# Seed mirror validation — 2026-09-05

The user confirmed complete textured DamagedHelmet rendering after TRUEOS fixed
the L3 allocation register address. The 21:55:28 screenshot and successful PBR
run support authored UVs/base color. All five images are processed by the
combined shader; independent material-map conformance remains unverified.

Only eight `picasso_tracking_v1` rows changed, all in revision 5:

| Records | Included | Tested | Fully respected |
| --- | --- | --- | --- |
| `r/5/accessor/3`, `r/5/buffer_view/3` — authored UV | yes | yes | yes |
| `r/5/buffer_view/4` — base-color image | yes | yes | yes |
| `r/5/buffer_view/5..8` — MR, emissive, AO, normal images | yes | yes, combined path | no |
| `r/5/buffer/0` — containing geometry/image payload | yes | yes, combined path | no |

No imported records or asset bytes changed: all 91 normalized records, 124 blob
chunks, metadata and other tracking rows retain their prior values. No general
scene, skinning, animation, alpha-mode or lighting-conformance claims were added.

Database SHA-256:

- Before: `a88715d592c9e7ba496168dbd1bf6ac72c65b4083e43b75806bf81d66c5e498e`
- After: `0b82f9bfcf0544397a93d0404c1fef7302e3c858401a00f2284337830a9d8d89`

Exact backup, row update plan, table hashes and receipt:
`../TRUEOS/bld/picasso-pbr-validation/success-2026-09-05/`.
See [renderer validation](../TRUEOS/tools/picasso-retained-texture-bake/VALIDATION.md)
for kernel/shader identity and the scope of the successful result.
