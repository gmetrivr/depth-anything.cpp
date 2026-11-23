# Store Video Capture Guidelines (`dg2n store scan`)

This guide defines the Standard Operating Procedure (SOP) for recording walk-through videos of retail stores so `dg2n-video-to-splat` produces high-density, drift-free 3D voxel and point-cloud reconstructions.

---

## 1. Camera & Phone Settings

| Setting | Recommended | Why |
|---|---|---|
| **Lens** | **`0.5x` Ultra-Wide** | Captures both sides of an aisle, floor base, and ceiling grid in a single frame. `1x` portrait sees mostly blank floor and ceiling, starving ICP of rigid 3D geometry. |
| **Orientation** | **Landscape (`16:9`)** | Maximizes horizontal field-of-view across left and right fixtures simultaneously. |
| **Resolution & FPS** | **`1080p` or `4K` @ `30 fps` or `60 fps`** | High shutter speed reduces motion blur; the pipeline downsamples to 640px @ `4–6 fps` with sharp frames. |
| **Stabilization** | **Action / Standard Stabilization ON** (or 3-axis gimbal) | Eliminates footsteps bobbing and rolling-shutter wobble. |
| **Auto-Exposure / Focus** | **AE/AF Lock** on a mid-lit aisle | Prevents brightness jumps between dark shelves and bright backlit signage. |

---

## 2. Camera Posture & Walking Technique

1. **Keep the Horizon Level (`0°` Pitch, `0°` Roll):**
   - Hold the phone at chest or eye height (`~1.3–1.5 m` above the floor) pointing **straight ahead** parallel to the floor.
   - **Never tilt the camera down at the floor or up at the ceiling** while walking—tilting introduces vertical pitch ambiguity and starves cross-view overlap.
2. **Walk at a Steady, Moderate Pace (`~0.7–1.0 m/s`):**
   - Each 12-frame window spans `2 seconds` at `6 fps`. Walking at `~0.8 m/s` gives a clean `~1.6 m` stereo baseline per window with `>75%` visual overlap between consecutive frames.
3. **Take Wide Arcs Around Aisle Ends (Never Pivot in Place):**
   - Rotating 90°–180° while standing still produces **zero camera translation (zero parallax)**, causing monocular scale and heading to slip.
   - Always walk a smooth, wide curve around end-caps while turning, and pan slowly (`<= 25°/second`).
4. **Keep `>= 1.2 m` Clearance from Fixtures:**
   - Avoid walking within `< 0.5 m` of a single rack or blank wall where a single textureless surface fills 100% of the frame.

---

## 3. Route Planning & Loop Closure

1. **Target `60–120 Seconds` per Zone / Section Clip (for Maximum Voxel Density):**
   - A 60–90 second clip (`360–540 frames @ 6 fps`) allocates **~20,000–30,000 points per frame** into a single zone, yielding crisp `700–1200` voxel-resolution shelves and fixtures with negligible odometry drift.
   - For large stores (`20–45 minutes` total), record **one 60–120 second loop per department/zone** rather than a single 45-minute take.
2. **Always Close Loops in the Same Viewing Direction:**
   - Visual loop-closure detection (`loop_close=1`) matches keyframes when you return to a previously visited landmark **facing the same direction** (within `~20°`).
   - Start each clip facing a distinct anchor landmark (e.g., main entrance signage or central feature island), walk the loop around the zone's aisles, and finish facing the **exact same landmark from the same spot** for `2–3 seconds`.
3. **Avoid Moving Obstacles:**
   - Film during low-traffic hours so shoppers or carts do not move across the frame during multi-view window solving.

---

## 4. Recommended `dg2n store scan` Bake Settings

- **Compute:** `GPU (Auto)`
- **Model:** `DA3 Base — default` (fast) or `DA3 Large — quality`
- **Output:** `Voxels`
- **Max frames:** `180–360` (for a 30–60s zone clip @ `6 fps`)
- **Window / Overlap:** `Window = 12`, `Overlap = 3`
- **Voxel res:** `700–1000` (with `Surface fusion` unchecked for direct dense voxels, or `Surface fusion` checked at `Fuse voxel = 0.10%–0.12%` for TSDF-merged surfaces)
- **De-ghosting:** Check **`ICP seams (B)`** and **`Loop closure (C)`** (when the clip returns to its starting landmark)
