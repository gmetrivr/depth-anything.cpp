#include "stream.hpp"

#include "engine.hpp"
#include "preprocess.hpp"
#include "image_io.hpp"
#include "reconstruct.hpp"
#include "sim3.hpp"
#include "icp.hpp"
#include "posegraph.hpp"
#include "spatial_hash.hpp"
#include "nested.hpp"
#include "common.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>

namespace da {

// Lightweight wall-clock accumulator for the phase-timing telemetry below.
using da_clock = std::chrono::steady_clock;
static inline double secs_since(std::chrono::time_point<da_clock> t0) {
    return std::chrono::duration<double>(da_clock::now() - t0).count();
}

// Dense per-pixel world points (in a window's LOCAL frame) for a handful of
// overlap frames, kept around to align the NEXT window against.
struct OverlapCache {
    int H = 0, W = 0, n = 0;          // n cached frames (== overlap of a full window)
    std::vector<uint8_t> valid;       // n*H*W : finite,d>0,conf>=thr at snapshot time
    std::vector<double>  xyz;         // n*H*W*3 : LOCAL-frame world points
    std::vector<float>   conf;        // n*H*W
    Sim3 G;                           // LOCAL -> GLOBAL for the cached window
};

// Per-window data retained for deferred emit + loop closure. Points are kept in
// each window's LOCAL frame so the final global transform (possibly corrected by
// pose-graph optimization) can be applied in one pass at the end.
struct WindowRec {
    Sim3 G, S_rel;                    // sequential global pose; relative seam (prev -> this)
    std::vector<float>   lx;          // LOCAL emit points, 3*n
    std::vector<uint8_t> lc;          // rgb, 3*n
    std::vector<float>   lr;          // base radius (pre-scale), n
    std::vector<int>     gf;          // global input-frame index per point, n
    std::vector<double>  key;         // downsampled LOCAL cloud for loop alignment, 3*m
    std::vector<double>  ccs, vds;    // per-frame camera centre / view dir (LOCAL), 3*nf interleaved
    std::vector<float>   vdesc;       // normalized visual descriptor of middle frame for drift-free loop proposal
    int mid = 0;                      // global index of this window's representative frame
    int w0 = 0;                       // global index of this window's FIRST frame
};

static std::vector<float> compute_visual_desc(const uint8_t* rgb, int H, int W) {
    if (!rgb || H < 8 || W < 8) return {};
    // 4x4 spatial grid x 3 channels x 8 colour bins (384 dims) + 8x8 downsampled RGB patch (192 dims)
    std::vector<float> d(384 + 192, 0.f);
    for (int y = 0; y < H; ++y) {
        int cy = std::min(3, (y * 4) / H);
        int py = std::min(7, (y * 8) / H);
        for (int x = 0; x < W; ++x) {
            int cx = std::min(3, (x * 4) / W);
            int px = std::min(7, (x * 8) / W);
            const uint8_t* p = rgb + ((size_t)y * W + x) * 3;
            int cell = (cy * 4 + cx) * 24;
            d[cell + 0 + (p[0] >> 5)] += 1.f;
            d[cell + 8 + (p[1] >> 5)] += 1.f;
            d[cell + 16 + (p[2] >> 5)] += 1.f;
            int poff = 384 + (py * 8 + px) * 3;
            d[poff + 0] += p[0] * (1.f / 255.f);
            d[poff + 1] += p[1] * (1.f / 255.f);
            d[poff + 2] += p[2] * (1.f / 255.f);
        }
    }
    auto norm_slice = [&](size_t a, size_t b) {
        double m = 0;
        for (size_t i = a; i < b; ++i) m += d[i];
        m /= (double)(b - a);
        double n2 = 0;
        for (size_t i = a; i < b; ++i) { d[i] -= (float)m; n2 += (double)d[i] * d[i]; }
        double inv = 1.0 / (std::sqrt(n2) + 1e-12);
        for (size_t i = a; i < b; ++i) d[i] = (float)(d[i] * inv * 0.70710678);
    };
    norm_slice(0, 384);
    norm_slice(384, 384 + 192);
    return d;
}

template <typename T>
static void write_pod(std::ostream& os, const T& v) {
    os.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
template <typename T>
static bool read_pod(std::istream& is, T& v) {
    return (bool)is.read(reinterpret_cast<char*>(&v), sizeof(T));
}
template <typename T>
static void write_vec(std::ostream& os, const std::vector<T>& v) {
    uint64_t n = (uint64_t)v.size();
    write_pod(os, n);
    if (n > 0) os.write(reinterpret_cast<const char*>(v.data()), (std::streamsize)(n * sizeof(T)));
}
template <typename T>
static bool read_vec(std::istream& is, std::vector<T>& v) {
    uint64_t n = 0;
    if (!read_pod(is, n)) return false;
    v.resize((size_t)n);
    if (n > 0 && !is.read(reinterpret_cast<char*>(v.data()), (std::streamsize)(n * sizeof(T)))) return false;
    return true;
}

static bool save_stream_ckpt(const std::string& path, int next_w0, int F, int chunk, int overlap,
                             const Sim3& G, bool have_prev, const OverlapCache& prev,
                             const std::vector<WindowRec>& wins) {
    if (path.empty()) return false;
    std::string tmp = path + ".tmp";
    std::ofstream os(tmp, std::ios::binary | std::ios::trunc);
    if (!os) return false;
    const uint32_t magic = 0x44415343u; // "DASC"
    write_pod(os, magic);
    write_pod(os, next_w0); write_pod(os, F); write_pod(os, chunk); write_pod(os, overlap);
    write_pod(os, G);
    uint8_t hp = have_prev ? 1 : 0; write_pod(os, hp);
    write_pod(os, prev.H); write_pod(os, prev.W); write_pod(os, prev.n); write_pod(os, prev.G);
    write_vec(os, prev.valid); write_vec(os, prev.xyz); write_vec(os, prev.conf);
    uint64_t nw = (uint64_t)wins.size(); write_pod(os, nw);
    for (const auto& r : wins) {
        write_pod(os, r.G); write_pod(os, r.S_rel); write_pod(os, r.mid); write_pod(os, r.w0);
        write_vec(os, r.lx); write_vec(os, r.lc); write_vec(os, r.lr); write_vec(os, r.gf);
        write_vec(os, r.key); write_vec(os, r.ccs); write_vec(os, r.vds); write_vec(os, r.vdesc);
    }
    os.close();
    if (!os) return false;
    return std::rename(tmp.c_str(), path.c_str()) == 0;
}

static bool load_stream_ckpt(const std::string& path, int F, int chunk, int overlap,
                             int& next_w0, Sim3& G, bool& have_prev, OverlapCache& prev,
                             std::vector<WindowRec>& wins) {
    if (path.empty()) return false;
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;
    uint32_t magic = 0; int cF = 0, cChunk = 0, cOverlap = 0;
    if (!read_pod(is, magic) || magic != 0x44415343u) return false;
    if (!read_pod(is, next_w0) || !read_pod(is, cF) || !read_pod(is, cChunk) || !read_pod(is, cOverlap)) return false;
    if (cF != F || cChunk != chunk || cOverlap != overlap || next_w0 <= 0 || next_w0 >= F) return false;
    if (!read_pod(is, G)) return false;
    uint8_t hp = 0; if (!read_pod(is, hp)) return false; have_prev = (hp != 0);
    if (!read_pod(is, prev.H) || !read_pod(is, prev.W) || !read_pod(is, prev.n) || !read_pod(is, prev.G)) return false;
    if (!read_vec(is, prev.valid) || !read_vec(is, prev.xyz) || !read_vec(is, prev.conf)) return false;
    uint64_t nw = 0; if (!read_pod(is, nw)) return false;
    wins.resize((size_t)nw);
    for (size_t i = 0; i < (size_t)nw; ++i) {
        auto& r = wins[i];
        if (!read_pod(is, r.G) || !read_pod(is, r.S_rel) || !read_pod(is, r.mid) || !read_pod(is, r.w0)) return false;
        if (!read_vec(is, r.lx) || !read_vec(is, r.lc) || !read_vec(is, r.lr) || !read_vec(is, r.gf)) return false;
        if (!read_vec(is, r.key) || !read_vec(is, r.ccs) || !read_vec(is, r.vds) || !read_vec(is, r.vdesc)) return false;
    }
    return true;
}

static void emit_wins_to_cloud(const std::vector<WindowRec>& wins, int F, int overlap, StreamCloud& out) {
    const int Wn = (int)wins.size();
    out.xyz.clear(); out.rgb.clear(); out.radius.clear();
    out.counts.assign(F, 0);
    out.window_pos.resize((size_t)3*Wn); out.window_mid_frame.resize(Wn);
    out.frame_pos.assign((size_t)3*F, 0.f); out.frame_fwd.assign((size_t)3*F, 0.f);
    for (int k = 0; k < Wn; ++k) {
        const WindowRec& r = wins[k];
        out.window_mid_frame[k] = r.mid;
        double gcpos[3];
        if (!r.ccs.empty()) { int mloc = (int)(r.ccs.size()/3)/2; sim3_apply(r.G, &r.ccs[3*mloc], gcpos); }
        else { gcpos[0]=r.G.t[0]; gcpos[1]=r.G.t[1]; gcpos[2]=r.G.t[2]; }
        out.window_pos[3*k+0]=(float)gcpos[0]; out.window_pos[3*k+1]=(float)gcpos[1]; out.window_pos[3*k+2]=(float)gcpos[2];

        const int nf = (int)(r.ccs.size()/3);
        const int lo = (r.w0 == 0) ? 0 : overlap;
        for (int i = lo; i < nf; ++i) {
            const int g = r.w0 + i;
            if (g < 0 || g >= F) continue;
            double gp[3]; sim3_apply(r.G, &r.ccs[3*i], gp);
            const double* vd = &r.vds[3*i];
            double d[3] = { r.G.R[0]*vd[0]+r.G.R[1]*vd[1]+r.G.R[2]*vd[2],
                            r.G.R[3]*vd[0]+r.G.R[4]*vd[1]+r.G.R[5]*vd[2],
                            r.G.R[6]*vd[0]+r.G.R[7]*vd[1]+r.G.R[8]*vd[2] };
            double dn = std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]) + 1e-30;
            out.frame_pos[3*g+0]=(float)gp[0]; out.frame_pos[3*g+1]=(float)gp[1]; out.frame_pos[3*g+2]=(float)gp[2];
            out.frame_fwd[3*g+0]=(float)(d[0]/dn); out.frame_fwd[3*g+1]=(float)(d[1]/dn); out.frame_fwd[3*g+2]=(float)(d[2]/dn);
        }
        const int n = (int)(r.lr.size());
        for (int i = 0; i < n; ++i) {
            double pL[3] = { r.lx[3*i+0], r.lx[3*i+1], r.lx[3*i+2] }, pG[3];
            sim3_apply(r.G, pL, pG);
            out.xyz.push_back((float)pG[0]); out.xyz.push_back((float)pG[1]); out.xyz.push_back((float)pG[2]);
            out.rgb.push_back(r.lc[3*i+0]); out.rgb.push_back(r.lc[3*i+1]); out.rgb.push_back(r.lc[3*i+2]);
            float rr = r.lr[i] * (float)r.G.s;
            if (!(rr > 0.f) || !std::isfinite(rr)) rr = 1e-4f;
            out.radius.push_back(rr);
            int gframe = r.gf[i];
            if (gframe >= 0 && gframe < F) out.counts[gframe]++;
        }
    }
}

// Back-project one frame's pixels into LOCAL world space, filling dense valid/xyz/conf.
// Validity is GEOMETRIC only (finite, d>0) — alignment uses conf as a weight, not a
// hard gate, so the overlap correspondence isn't starved at low-confidence window
// edges. conf is stored raw (or 1.0 when the model gives no conf) for that weighting.
static void backproject_frame_dense(const ViewResult& view,
                                     const std::array<float,9>& K,
                                     const std::array<float,16>& E4,
                                     int H, int W, bool have_conf,
                                     uint8_t* valid, double* xyz, float* conf) {
    std::array<float,9> Kinv; std::array<float,16> c2w;
    const bool kok = inv3(K, Kinv);
    const bool eok = inv4(E4, c2w);
    double ki[9]; for (int k=0;k<9;++k) ki[k] = kok ? (double)Kinv[k] : 0.0;
    double cw[16]; for (int k=0;k<16;++k) cw[k] = eok ? (double)c2w[k] : 0.0;
    const size_t plane = (size_t)H*(size_t)W;
    const float* dF = view.depth.data();
    const float* cF = (have_conf && view.conf.size()==plane) ? view.conf.data() : nullptr;
    for (int v = 0; v < H; ++v) {
        for (int u = 0; u < W; ++u) {
            size_t pix = (size_t)v*W + u;
            float d = dF[pix];
            float cc = cF ? cF[pix] : 1.0f;
            conf[pix] = cc;
            bool ok = kok && eok && std::isfinite(d) && d > 0.0f;
            if (!ok) { valid[pix] = 0; xyz[3*pix+0]=xyz[3*pix+1]=xyz[3*pix+2]=0.0; continue; }
            double rx = ki[0]*u + ki[1]*v + ki[2];
            double ry = ki[3]*u + ki[4]*v + ki[5];
            double rz = ki[6]*u + ki[7]*v + ki[8];
            double xc = rx*d, yc = ry*d, zc = rz*d;
            xyz[3*pix+0] = cw[0]*xc + cw[1]*yc + cw[2]*zc + cw[3];
            xyz[3*pix+1] = cw[4]*xc + cw[5]*yc + cw[6]*zc + cw[7];
            xyz[3*pix+2] = cw[8]*xc + cw[9]*yc + cw[10]*zc + cw[11];
            valid[pix] = 1;
        }
    }
}

// Camera centre and view direction (+z_cam) in the window LOCAL frame, from a
// world->cam [R|t] extrinsic (E4 row-major 4x4). Uses the same inv4() the back-
// projection relies on (c2w = inv(ext); its translation column is the camera
// centre, its 3rd rotation column is the +z_cam axis in world). Used for loop proposal.
struct CamCenterDir { std::array<double,3> cc, vd; };
static CamCenterDir cam_center_dir_local(const std::array<float,16>& E4) {
    CamCenterDir r;
    std::array<float,16> c2w;
    if (!inv4(E4, c2w)) { r.cc = {0,0,0}; r.vd = {0,0,1}; return r; }
    r.cc = { c2w[3], c2w[7], c2w[11] };            // camera centre in world
    r.vd = { c2w[2], c2w[6], c2w[10] };            // c2w * (0,0,1): +z_cam axis in world
    return r;
}

// Attempt a loop-closure measurement aligning window b's keyframe cloud onto
// window a's (both in their LOCAL frames). Correspondences are established in the
// frame of the current SEQUENTIAL relative estimate (G_a^{-1}∘G_b) with a TIGHT
// match gate, so only the genuinely-overlapping surface matches — a large gate in
// a small room would match across the room and fit a garbage transform. Returns
// the relative Sim3 Z (b-local -> a-local) and the ICP correspondence count; false
// when the true overlap is too thin (filters a mis-proposed candidate).
static bool measure_loop_with_seed(const WindowRec& a, const WindowRec& b, const Sim3& init0,
                                   const IcpParams& mp, double match_gate, int min_corr,
                                   Sim3& Z_out, int& corr_out) {
    const int nb = (int)(b.key.size()/3), na = (int)(a.key.size()/3);
    if (nb < min_corr || na < min_corr) return false;

    SpatialHash hash(a.key.data(), na, std::max(match_gate, 1e-3));
    std::vector<double> src, tgt, w; double d2;
    for (int i=0;i<nb;++i) {
        double ba[3]; sim3_apply(init0, &b.key[3*i], ba);        // b point in a's local frame
        int j = hash.nearest(ba, match_gate, d2);
        if (j < 0) continue;
        src.push_back(b.key[3*i]); src.push_back(b.key[3*i+1]); src.push_back(b.key[3*i+2]);
        tgt.push_back(a.key[3*j]); tgt.push_back(a.key[3*j+1]); tgt.push_back(a.key[3*j+2]);
        w.push_back(1.0);
    }
    const int M = (int)w.size();
    if (M < min_corr) return false;
    Sim3 S; double rms=0;
    if (!umeyama_sim3_weighted(src.data(), tgt.data(), w.data(), M, S, rms, min_corr)) return false;
    S.s = 1.0;   // lock scale: consecutive windows share scale via the chain, and a
                 // near-planar revisit can't observe scale (=> a spurious loop scale).

    // Fine point-to-plane ICP against a's surface (reuses task B).
    std::vector<double> na_normals;
    estimate_normals(a.key, mp.normal_radius, nullptr, na_normals);
    IcpResult ir = icp_point_to_plane(b.key, a.key, na_normals, S, mp);
    if (!ir.ok || ir.corr < min_corr) return false;
    Z_out = ir.T; corr_out = ir.corr;
    return true;
}

static bool measure_loop(const WindowRec& a, const WindowRec& b, const IcpParams& mp,
                         double match_gate, int min_corr, Sim3& Z_out, int& corr_out,
                         bool visual_seed = false) {
    // 1. Try the sequential relative pose seed (G_a^{-1} o G_b).
    Sim3 init0 = sim3_compose(sim3_inverse(a.G), b.G);
    if (measure_loop_with_seed(a, b, init0, mp, match_gate, min_corr, Z_out, corr_out))
        return true;
    // 2. When proposed by visual appearance similarity across a long drifted trajectory,
    //    both windows' local clouds are already in their own camera frames facing the
    //    same scene — seed with identity in local camera space.
    if (visual_seed) {
        Sim3 id;
        if (measure_loop_with_seed(a, b, id, mp, match_gate * 1.5, min_corr, Z_out, corr_out))
            return true;
    }
    return false;
}

// Pure sliding-window stitch/fuse. Pulls each window's fused views + color from
// `source`. Sequential seams are solved by weighted Umeyama (optionally refined
// by point-to-plane ICP, task B); when loop_close (task C) is set, revisits are
// detected, measured, and a Sim3 pose graph redistributes the drift before the
// single deferred emit pass.
bool stream_points_core(int nframes, const StreamParams& p,
                        const WindowSource& source, StreamCloud& out, std::string& err) {
    const int F = nframes;
    if (F == 0) { err = "stream: no frames"; return false; }

    int chunk = p.chunk_size < 2 ? 2 : p.chunk_size;
    int overlap = p.overlap;
    if (overlap < 0) overlap = 0;
    if (overlap > chunk - 1) overlap = chunk - 1;
    int step = chunk - overlap;
    if (step < 1) step = 1;
    double conf_pct = p.conf_pct; if (conf_pct < 0) conf_pct = 0; if (conf_pct > 100) conf_pct = 100;
    float point_size = (p.point_size > 0.f) ? p.point_size : 1.f;

    out.xyz.clear(); out.rgb.clear(); out.radius.clear();
    out.counts.assign(F, 0);
    out.warnings = 0; out.loops_found = 0; out.window_pos.clear();

    Sim3 G;                 // current window LOCAL -> GLOBAL
    OverlapCache prev;      // previous window's tail-overlap geometry
    bool have_prev = false;
    std::vector<WindowRec> wins;   // per-window records (deferred emit)
    int start_w0 = 0;
    if (!p.checkpoint_path.empty() &&
        load_stream_ckpt(p.checkpoint_path, F, chunk, overlap, start_w0, G, have_prev, prev, wins)) {
        DA_LOG("stream: resumed from checkpoint %s at frame %d/%d (%zu windows)",
               p.checkpoint_path.c_str(), start_w0, F, wins.size());
        if (p.on_progress) p.on_progress(start_w0, F);
    }

    // Phase-timing telemetry. Separates GPU inference (source()) from the
    // single-threaded CPU geometry (stitch/ICP, loop closure, emit) so we can
    // see where wall-clock actually goes and whether the GPU is starved.
    double t_infer = 0, t_stitch = 0, t_loop = 0, t_emit = 0;

    for (int w0 = start_w0; w0 < F; w0 += step) {
        const int w1 = std::min(w0 + chunk, F);
        const int nv = w1 - w0;

        // --- pull this window's fused views + color from the source ---
        std::vector<ViewResult> views;
        std::vector<std::vector<uint8_t>> rgb_store;
        int H = 0, W = 0;
        auto t_src = da_clock::now();
        if (!source(w0, w1, views, rgb_store, H, W, err)) return false;
        t_infer += secs_since(t_src);
        auto t_cpu = da_clock::now();
        if ((int)views.size() != nv || (int)rgb_store.size() != nv) {
            err = "stream: window source returned wrong view/color count"; return false; }
        const size_t plane = (size_t)H * (size_t)W;

        // K, E4, color pointers per view.
        std::vector<std::array<float,9>>  K(nv);
        std::vector<std::array<float,16>> E4(nv);
        std::vector<const uint8_t*>       images_u8(nv);
        bool have_conf = true;
        std::vector<float> conf_all; conf_all.reserve((size_t)nv*plane);
        for (int i = 0; i < nv; ++i) {
            K[i] = views[i].intr;
            std::array<float,16> e4{}; for (int k=0;k<12;++k) e4[k]=views[i].ext[k];
            e4[12]=0.f; e4[13]=0.f; e4[14]=0.f; e4[15]=1.f; E4[i]=e4;
            if (rgb_store[i].size() != plane*3) {
                err = "stream: window source color size mismatch"; return false; }
            images_u8[i] = rgb_store[i].data();
            if (views[i].conf.size() == plane) conf_all.insert(conf_all.end(), views[i].conf.begin(), views[i].conf.end());
            else have_conf = false;
        }
        if (!have_conf) conf_all.clear();
        float conf_thr = -1e30f;
        if (have_conf && !conf_all.empty()) conf_thr = (float)percentile_linear(conf_all, conf_pct);

        // --- stitch: solve G (LOCAL -> GLOBAL) for this window ---
        Sim3 S_rel;   // relative seam prev -> this (identity for window 0)
        if (!have_prev) {
            G = Sim3();  // window 0 defines the global frame
        } else {
            const int ov = std::min(prev.n, nv);
            std::vector<double> srcv, tgtv, wv;  // cur-local -> prev-local
            srcv.reserve((size_t)ov*plane*3);
            for (int o = 0; o < ov; ++o) {
                std::array<float,9> Kinv; std::array<float,16> c2w;
                if (!inv3(K[o], Kinv) || !inv4(E4[o], c2w)) continue;
                double ki[9]; for (int k=0;k<9;++k) ki[k]=(double)Kinv[k];
                double cw[16]; for (int k=0;k<16;++k) cw[k]=(double)c2w[k];
                const float* dcur = views[o].depth.data();
                const float* ccur = (have_conf && views[o].conf.size()==plane) ? views[o].conf.data() : nullptr;
                const uint8_t* pvalid = prev.valid.data() + (size_t)o*plane;
                const double*  pxyz   = prev.xyz.data()   + (size_t)o*plane*3;
                const float*   pconf  = prev.conf.data()  + (size_t)o*plane;
                for (int v = 0; v < H; ++v) for (int u = 0; u < W; ++u) {
                    size_t pix = (size_t)v*W + u;
                    if (!pvalid[pix]) continue;
                    float d = dcur[pix];
                    if (!std::isfinite(d) || d <= 0.0f) continue;
                    float cc = ccur ? ccur[pix] : 1.0f;   // weight only, no hard gate
                    double rx = ki[0]*u + ki[1]*v + ki[2];
                    double ry = ki[3]*u + ki[4]*v + ki[5];
                    double rz = ki[6]*u + ki[7]*v + ki[8];
                    double xc = rx*d, yc = ry*d, zc = rz*d;
                    srcv.push_back(cw[0]*xc + cw[1]*yc + cw[2]*zc + cw[3]);
                    srcv.push_back(cw[4]*xc + cw[5]*yc + cw[6]*zc + cw[7]);
                    srcv.push_back(cw[8]*xc + cw[9]*yc + cw[10]*zc + cw[11]);
                    tgtv.push_back(pxyz[3*pix+0]); tgtv.push_back(pxyz[3*pix+1]); tgtv.push_back(pxyz[3*pix+2]);
                    double wgt = std::min((double)cc, (double)pconf[pix]);
                    wv.push_back(wgt > 0.0 ? wgt : 1e-6);
                }
            }
            const int M = (int)wv.size();
            double rms = 0.0;
            bool ok = M >= p.min_overlap_pts &&
                      umeyama_sim3_weighted(srcv.data(), tgtv.data(), wv.data(), M, S_rel, rms, p.min_overlap_pts);
            if (!ok) {
                out.warnings++;
                DA_LOG("stream: window @%d degenerate stitch (M=%d) -> identity seam", w0, M);
                S_rel = Sim3();
            } else {
                // On long sequences (F > 100) without metric branch, guard against
                // cumulative Sim3 scale drift (G.s collapsing toward 0 or blowing up
                // over hundreds of windows) while preserving the overlap centroid match
                // before point-to-plane ICP refinement.
                if (F > 100 && !p.metric && S_rel.s > 0.0) {
                    double s_old = S_rel.s;
                    double s_new = std::max(0.85, std::min(1.18, s_old));
                    double g_next = prev.G.s * s_new;
                    if (g_next < 0.5) s_new = 0.5 / prev.G.s;
                    if (g_next > 2.0) s_new = 2.0 / prev.G.s;
                    if (std::fabs(s_new - s_old) > 1e-6) {
                        double wsum = 0.0, ms[3] = {0, 0, 0};
                        for (int i = 0; i < M; ++i) {
                            double wi = wv[i];
                            wsum += wi;
                            ms[0] += wi * srcv[3*i+0];
                            ms[1] += wi * srcv[3*i+1];
                            ms[2] += wi * srcv[3*i+2];
                        }
                        if (wsum > 0.0) {
                            ms[0] /= wsum; ms[1] /= wsum; ms[2] /= wsum;
                            double rmsrc[3] = {
                                S_rel.R[0]*ms[0] + S_rel.R[1]*ms[1] + S_rel.R[2]*ms[2],
                                S_rel.R[3]*ms[0] + S_rel.R[4]*ms[1] + S_rel.R[5]*ms[2],
                                S_rel.R[6]*ms[0] + S_rel.R[7]*ms[1] + S_rel.R[8]*ms[2]
                            };
                            double ds = s_old - s_new;
                            S_rel.t[0] += ds * rmsrc[0];
                            S_rel.t[1] += ds * rmsrc[1];
                            S_rel.t[2] += ds * rmsrc[2];
                            S_rel.s = s_new;
                        }
                    }
                }
                DA_LOG("stream: window @%d stitch M=%d s=%.4f rms=%.4g", w0, M, S_rel.s, rms);
                if (p.icp_refine) {
                    // Scale the ICP neighbourhood/gates to the overlap's OWN extent —
                    // monocular reconstructions live at an arbitrary metric scale, so
                    // absolute radii would either no-op or engulf the whole overlap
                    // (O(N^2)). Also downsample: a rigid seam fit needs ~1e4 points,
                    // not the full dense overlap.
                    double lo[3]={srcv[0],srcv[1],srcv[2]}, hi[3]={srcv[0],srcv[1],srcv[2]};
                    for (int i=0;i<M;++i) for(int k=0;k<3;++k){ lo[k]=std::min(lo[k],tgtv[3*i+k]); hi[k]=std::max(hi[k],tgtv[3*i+k]); }
                    double odiag=std::sqrt((hi[0]-lo[0])*(hi[0]-lo[0])+(hi[1]-lo[1])*(hi[1]-lo[1])+(hi[2]-lo[2])*(hi[2]-lo[2]));
                    int ds = std::max(1, M / 40000);
                    std::vector<double> s2, t2; s2.reserve((size_t)3*(M/ds+1)); t2.reserve((size_t)3*(M/ds+1));
                    for (int i=0;i<M;i+=ds) for(int k=0;k<3;++k){ s2.push_back(srcv[3*i+k]); t2.push_back(tgtv[3*i+k]); }
                    IcpParams ip = p.icp;
                    if (odiag > 0){ ip.normal_radius=0.02*odiag; ip.max_corr_dist=0.03*odiag; ip.huber_delta=0.015*odiag; }
                    std::vector<double> tn;
                    estimate_normals(t2, ip.normal_radius, nullptr, tn);
                    IcpResult irr = icp_point_to_plane(s2, t2, tn, S_rel, ip);
                    if (irr.ok && irr.rms_after <= irr.rms_before) {
                        S_rel = irr.T;
                        DA_LOG("stream: window @%d icp rank=%d iters=%d rms %.4g->%.4g",
                               w0, irr.rank, irr.iters, irr.rms_before, irr.rms_after);
                    }
                }
            }
            G = sim3_compose(prev.G, S_rel);  // cur-local -> global
        }

        // --- collect NEW frames' points in LOCAL frame (deferred emit) ---
        WindowRec rec; rec.G = G; rec.S_rel = S_rel; rec.mid = w0 + nv/2; rec.w0 = w0;
        rec.ccs.resize((size_t)3*nv); rec.vds.resize((size_t)3*nv);
        for (int i = 0; i < nv; ++i) {
            CamCenterDir cd = cam_center_dir_local(E4[i]);
            rec.ccs[3*i+0]=cd.cc[0]; rec.ccs[3*i+1]=cd.cc[1]; rec.ccs[3*i+2]=cd.cc[2];
            rec.vds[3*i+0]=cd.vd[0]; rec.vds[3*i+1]=cd.vd[1]; rec.vds[3*i+2]=cd.vd[2];
        }
        const int new_lo = have_prev ? std::min(prev.n, nv) : 0;
        const int newN = nv - new_lo;
        if (newN > 0) {
            std::vector<float> depth_slice; depth_slice.reserve((size_t)newN*plane);
            std::vector<float> conf_slice;  if (have_conf) conf_slice.reserve((size_t)newN*plane);
            std::vector<std::array<float,9>>  Ks(newN);
            std::vector<std::array<float,16>> Es(newN);
            std::vector<const uint8_t*>       Iu8(newN);
            for (int i = new_lo; i < nv; ++i) {
                int j = i - new_lo;
                depth_slice.insert(depth_slice.end(), views[i].depth.begin(), views[i].depth.end());
                if (have_conf) conf_slice.insert(conf_slice.end(), views[i].conf.begin(), views[i].conf.end());
                Ks[j] = K[i]; Es[j] = E4[i]; Iu8[j] = images_u8[i];
            }
            WorldPoints wp = back_project(depth_slice, have_conf ? conf_slice : std::vector<float>{},
                                          Ks, Es, Iu8, H, W, newN, conf_thr);
            const size_t np = wp.frame.size();

            // Per-window budget: uniform stride over the frame-major point list.
            size_t stride = 1;
            if (p.global_budget > 0 && F > 0) {
                double quota = (double)p.global_budget * (double)newN / (double)F;
                if (quota < 1.0) quota = 1.0;
                if ((double)np > quota) stride = (size_t)std::ceil((double)np / quota);
                if (stride < 1) stride = 1;
            }

            for (size_t k = 0; k < np; k += stride) {
                int f = wp.frame[k], u = wp.u[k], v = wp.v[k];
                rec.lx.push_back(wp.xyz[3*k+0]); rec.lx.push_back(wp.xyz[3*k+1]); rec.lx.push_back(wp.xyz[3*k+2]);
                rec.lc.push_back(wp.rgb[3*k+0]); rec.lc.push_back(wp.rgb[3*k+1]); rec.lc.push_back(wp.rgb[3*k+2]);
                float d = depth_slice[(size_t)f*plane + (size_t)v*W + u];
                float fx = Ks[f][0], fy = Ks[f][4];
                float rr_base = 0.5f * (d/fx + d/fy) * point_size;   // scale applied at emit
                rec.lr.push_back(rr_base);
                rec.gf.push_back(w0 + new_lo + f);
            }
        }
        // downsampled LOCAL keyframe cloud for loop alignment (~3k points).
        if (p.loop_close) {
            const int n = (int)(rec.lx.size()/3);
            int ks = std::max(1, n / 3000);
            rec.key.reserve((size_t)3 * (n/ks + 1));
            for (int i = 0; i < n; i += ks) {
                rec.key.push_back(rec.lx[3*i]); rec.key.push_back(rec.lx[3*i+1]); rec.key.push_back(rec.lx[3*i+2]);
            }
            if (!rgb_store.empty()) {
                rec.vdesc = compute_visual_desc(rgb_store[nv/2].data(), H, W);
            }
        }
        wins.push_back(std::move(rec));

        // --- snapshot this window's TAIL overlap (LOCAL frame) for the next stitch ---
        const int ovkeep = std::min(overlap, nv);
        prev.H = H; prev.W = W; prev.n = ovkeep; prev.G = G;
        prev.valid.assign((size_t)ovkeep*plane, 0);
        prev.xyz.assign((size_t)ovkeep*plane*3, 0.0);
        prev.conf.assign((size_t)ovkeep*plane, 0.f);
        for (int o = 0; o < ovkeep; ++o) {
            int vi = nv - ovkeep + o;  // local index of this tail frame
            backproject_frame_dense(views[vi], K[vi], E4[vi], H, W, have_conf,
                                    prev.valid.data() + (size_t)o*plane,
                                    prev.xyz.data()   + (size_t)o*plane*3,
                                    prev.conf.data()  + (size_t)o*plane);
        }
        have_prev = true;

        t_stitch += secs_since(t_cpu);
        if (p.on_progress) p.on_progress(w1, F);
        const int ckpt_every = p.checkpoint_every > 0 ? p.checkpoint_every : 20;
        if (w1 < F && ((int)wins.size() % ckpt_every == 0)) {
            if (!p.checkpoint_path.empty()) {
                save_stream_ckpt(p.checkpoint_path, w0 + step, F, chunk, overlap, G, have_prev, prev, wins);
            }
            if (p.on_checkpoint) {
                StreamCloud partial;
                emit_wins_to_cloud(wins, w1, overlap, partial);
                p.on_checkpoint(w1, F, partial);
            }
        }
        if (w1 >= F) break;  // last (possibly truncated) window processed
    }
    if (!p.checkpoint_path.empty()) {
        std::remove(p.checkpoint_path.c_str());
    }

    // --- task C: detect + measure loops, optimize the Sim3 pose graph ---
    auto t_loop0 = da_clock::now();
    const int Wn = (int)wins.size();
    if (p.loop_close && Wn > p.loop_min_gap) {
        PoseGraph pg;
        pg.nodes.resize(Wn);
        for (int k = 0; k < Wn; ++k) pg.nodes[k] = wins[k].G;
        pg.fixed.assign(Wn, 0); pg.fixed[0] = 1;
        for (int k = 1; k < Wn; ++k)
            pg.edges.push_back(PoseEdge{ k-1, k, wins[k].S_rel, 1.0, false });

        // Per-frame global camera centres/dirs under the sequential estimate. A loop
        // is PROPOSED when ANY frame of window a sits near an ANY frame of window b
        // with an aligned view direction (captures wrap-around overlap, where the
        // window centroids are far apart but the frame ranges meet).
        std::vector<std::vector<double>> gc(Wn), gd(Wn);   // per-window 3*nf interleaved
        for (int k = 0; k < Wn; ++k) {
            const Sim3& g = wins[k].G; const int nf = (int)(wins[k].ccs.size()/3);
            gc[k].resize((size_t)3*nf); gd[k].resize((size_t)3*nf);
            for (int i = 0; i < nf; ++i) {
                double gcpos[3]; sim3_apply(g, &wins[k].ccs[3*i], gcpos);
                gc[k][3*i+0]=gcpos[0]; gc[k][3*i+1]=gcpos[1]; gc[k][3*i+2]=gcpos[2];
                const double* vd = &wins[k].vds[3*i];
                double d[3] = { g.R[0]*vd[0]+g.R[1]*vd[1]+g.R[2]*vd[2],
                                g.R[3]*vd[0]+g.R[4]*vd[1]+g.R[5]*vd[2],
                                g.R[6]*vd[0]+g.R[7]*vd[1]+g.R[8]*vd[2] };
                double n = std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]) + 1e-30;
                gd[k][3*i+0]=d[0]/n; gd[k][3*i+1]=d[1]/n; gd[k][3*i+2]=d[2]/n;
            }
        }
        auto windows_overlap = [&](int a, int b) -> bool {
            const double pt2 = p.loop_pos_thr * p.loop_pos_thr;
            const int na=(int)(gc[a].size()/3), nb=(int)(gc[b].size()/3);
            for (int ia=0; ia<na; ++ia) for (int ib=0; ib<nb; ++ib) {
                double dx=gc[a][3*ia]-gc[b][3*ib], dy=gc[a][3*ia+1]-gc[b][3*ib+1], dz=gc[a][3*ia+2]-gc[b][3*ib+2];
                if (dx*dx+dy*dy+dz*dz > pt2) continue;
                if (gd[a][3*ia]*gd[b][3*ib]+gd[a][3*ia+1]*gd[b][3*ib+1]+gd[a][3*ia+2]*gd[b][3*ib+2] >= p.loop_dir_thr)
                    return true;
            }
            return false;
        };
        auto visual_overlap = [&](int a, int b) -> bool {
            const auto& va = wins[a].vdesc;
            const auto& vb = wins[b].vdesc;
            if (va.empty() || va.size() != vb.size()) return false;
            double dot = 0;
            for (size_t i = 0; i < va.size(); ++i) dot += (double)va[i] * vb[i];
            return dot >= 0.85;
        };
        IcpParams mp = p.icp; if (!(mp.max_corr_dist > 0)) mp.max_corr_dist = 0.2;
        for (int a = 0; a < Wn; ++a) for (int b = a + p.loop_min_gap; b < Wn; ++b) {
            const bool geom_ov = windows_overlap(a, b);
            const bool vis_ov  = !geom_ov && visual_overlap(a, b);
            if (!geom_ov && !vis_ov) continue;
            Sim3 Z; int corr = 0;
            if (!measure_loop(wins[a], wins[b], mp, p.loop_match_dist, p.loop_min_corr, Z, corr, vis_ov)) continue;
            pg.edges.push_back(PoseEdge{ a, b, Z, p.loop_info, true });
            out.loops_found++;
            DA_LOG("stream: loop closure %d<->%d (%s) corr=%d", a, b, vis_ov ? "visual" : "spatial", corr);
        }

        if (out.loops_found > 0) {
            PoseGraphResult pr = optimize_pose_graph(pg, p.pose_graph);
            DA_LOG("stream: pose graph loops=%d free=%d cost %.4g->%.4g",
                   out.loops_found, pr.free_dofs, pr.cost0, pr.cost);
            for (int k = 0; k < Wn; ++k) wins[k].G = pg.nodes[k];
        }
    }

    t_loop = secs_since(t_loop0);

    // --- deferred emit pass: transform each window's LOCAL points by its final G ---
    auto t_emit0 = da_clock::now();
    emit_wins_to_cloud(wins, F, overlap, out);
    t_emit = secs_since(t_emit0);
    DA_LOG("stream timing: infer(gpu)=%.2fs stitch(cpu)=%.2fs loop(cpu)=%.2fs emit(cpu)=%.2fs "
           "| windows=%d loops=%d pts=%zu", t_infer, t_stitch, t_loop, t_emit,
           Wn, out.loops_found, out.radius.size());

    if (out.radius.empty()) { err = "stream: no points survived (raise conf_pct or check parallax)"; return false; }
    return true;
}

// Production entry: build an Engine-backed WindowSource (load images -> fused
// multi-view inference -> processed-resolution color) and run the core.
bool stream_points(Engine& eng, const std::vector<std::string>& frame_paths,
                   const Config& cfg, const StreamParams& p,
                   StreamCloud& out, std::string& err) {
    const int F = (int)frame_paths.size();
    if (p.metric && !eng.is_nested()) {
        err = "stream: metric requested but model is not a nested (anyview+metric) engine"; return false; }

    double metric_scale_sum = 0.0; int metric_scale_n = 0;   // for out.metric_scale reporting
    WindowSource source = [&](int w0, int w1, std::vector<ViewResult>& views,
                              std::vector<std::vector<uint8_t>>& rgb,
                              int& H, int& W, std::string& e) -> bool {
        const int nv = w1 - w0;
        std::vector<Image> imgs(nv);
        for (int i = 0; i < nv; ++i)
            if (!load_image_rgb(frame_paths[w0+i], imgs[i])) {
                e = "stream: load image failed: " + frame_paths[w0+i]; return false; }
        H = 0; W = 0;
        if (!eng.depth_pose_multi(imgs, views, H, W) || (int)views.size() != nv) {
            e = "stream: depth_pose_multi failed"; return false; }
        rgb.assign(nv, {});
        for (int i = 0; i < nv; ++i) {
            Preprocessed pp;
            if (!preprocess_real(imgs[i], cfg, pp, &rgb[i]) || pp.H != H || pp.W != W) {
                e = "stream: preprocess color mismatch"; return false; }
        }

        // --- optional: rescale this window to METRES (nested metric branch) ---
        // The multi-view anyview depth above is relative (arbitrary scale). Run the
        // metric ViT-L branch on a few frames spread across the window (same 504px
        // preprocess_real, so pixel-aligned), fit each against the anyview depth via
        // the NestedAligner least-squares, and take the MEDIAN scale_factor. A single
        // frame's estimate is noisy (~30% CV), but the whole window is one scene at
        // one true scale, so the median is stable (~8% SE over ~6 samples). Scaling
        // depth AND the pose translation by the same K is a uniform similarity resize
        // of the window — geometry preserved, now in metres. Downstream stitch scales
        // then collapse toward 1 (all windows metric), so no core changes are needed.
        if (p.metric) {
            std::vector<double> scales;
            const int mstride = std::max(1, nv / 6);   // ~6 metric evals per window
            for (int i = 0; i < nv; i += mstride) {
                std::vector<float> mraw, sky; int Hm=0, Wm=0;
                if (!eng.metric_branch(imgs[i], mraw, sky, Hm, Wm) || Hm != H || Wm != W) continue;
                AnyviewOut any; any.depth = views[i].depth; any.depth_conf = views[i].conf;
                any.extrinsics = views[i].ext; any.intrinsics = views[i].intr;
                MetricOut m; m.depth = std::move(mraw); m.sky = std::move(sky);
                NestedOut no = NestedAligner::align(any, m, H, W);
                if (no.scale_factor > 0.f && std::isfinite(no.scale_factor)) scales.push_back(no.scale_factor);
            }
            if (scales.empty()) { e = "stream: metric branch produced no valid scale"; return false; }
            std::sort(scales.begin(), scales.end());
            const double K = scales[scales.size()/2];   // robust median window scale
            for (int i = 0; i < nv; ++i) {
                for (float& d : views[i].depth) d *= (float)K;
                views[i].ext[3] *= (float)K; views[i].ext[7] *= (float)K; views[i].ext[11] *= (float)K;
            }
            metric_scale_sum += K; metric_scale_n += 1;
            DA_LOG("stream: window @%d metric scale=%.4f (median of %zu samples)", w0, K, scales.size());
        }
        return true;
    };
    bool ok = stream_points_core(F, p, source, out, err);
    if (ok && metric_scale_n > 0) out.metric_scale = (float)(metric_scale_sum / metric_scale_n);
    return ok;
}

} // namespace da
