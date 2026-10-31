Leaf-cell dump: `evidence/r41/hbm_bw/resources/run1_routed/leaf_cells.tsv.gz`: placed leaf cells by SLR. LUT and LUTRAM columns count leaf cells (a LUT site holding two LUT cells, or a LUTRAM macro's RAMD32 pair, counts twice), so they exceed report_utilization's site-based LUT figures; FF, CARRY8, RAMB and DSP counts are exact.


#### SLR0

| Group | LUT cells | LUTRAM/SRL cells | FF | FF replicas (`_rep`) | CARRY8 | RAMB36 | RAMB18 | DSP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TNCA tile 0 | 268,740 | 17,070 | 65,345 | 1,559 | 9,246 | 32 | 15 | 171 |
| TNCA DCO/TMU replicas | 7,615 | 360 | 2,032 | 2,032 | 343 | 0 | 0 | 0 |
| TNCA outlets (line AXI + links) | 22,599 | 2,016 | 19,429 | 57 | 186 | 0 | 0 | 0 |
| TNCA mesh | 863 | 1,400 | 76 | 0 | 0 | 0 | 0 | 0 |
| TNCA other kernel logic | 1,197 | 1,448 | 785 | 9 | 72 | 32 | 0 | 0 |
| Platform: HBM subsystem (hmss) | 10,736 | 5,042 | 17,062 | 2,152 | 58 | 4 | 0 | 0 |
| Platform: other dynamic region (ulp) | 5,896 | 22 | 8,985 | 2 | 85 | 0 | 0 | 0 |
| Platform: static shell (blp and others) | 32,074 | 3,131 | 45,243 | 792 | 274 | 62 | 3 | 0 |
| **SLR total** | **349,720** | **30,489** | **158,957** | **6,603** | **10,264** | **130** | **18** | **171** |

#### SLR1

| Group | LUT cells | LUTRAM/SRL cells | FF | FF replicas (`_rep`) | CARRY8 | RAMB36 | RAMB18 | DSP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TNCA tile 1 | 269,183 | 17,078 | 65,382 | 1,603 | 9,247 | 32 | 15 | 171 |
| TNCA DCO/TMU replicas | 7,611 | 360 | 2,032 | 2,032 | 343 | 0 | 0 | 0 |
| TNCA outlets (line AXI + links) | 1,078 | 2,016 | 1,569 | 3 | 0 | 0 | 0 | 0 |
| TNCA mesh | 1,320 | 2,456 | 110 | 0 | 0 | 0 | 0 | 0 |
| TNCA other kernel logic | 7,040 | 5,792 | 6,388 | 28 | 261 | 32 | 0 | 7 |
| Platform: HBM subsystem (hmss) | 0 | 0 | 3,831 | 0 | 0 | 0 | 0 | 0 |
| Platform: other dynamic region (ulp) | 642 | 1 | 1,084 | 4 | 0 | 0 | 0 | 0 |
| Platform: static shell (blp and others) | 68,396 | 7,199 | 63,324 | 477 | 581 | 81 | 2 | 0 |
| **SLR total** | **355,270** | **34,902** | **143,720** | **4,147** | **10,432** | **145** | **17** | **178** |

#### SLR2

| Group | LUT cells | LUTRAM/SRL cells | FF | FF replicas (`_rep`) | CARRY8 | RAMB36 | RAMB18 | DSP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TNCA tile 2 | 267,932 | 17,078 | 65,303 | 1,571 | 9,220 | 32 | 15 | 171 |
| TNCA DCO/TMU replicas | 7,611 | 360 | 2,032 | 2,032 | 343 | 0 | 0 | 0 |
| TNCA mesh | 879 | 1,400 | 78 | 0 | 0 | 0 | 0 | 0 |
| TNCA other kernel logic | 432 | 1,448 | 610 | 2 | 15 | 32 | 0 | 0 |
| Platform: HBM subsystem (hmss) | 1,712 | 1,992 | 6,170 | 1,304 | 14 | 0 | 0 | 0 |
| Platform: other dynamic region (ulp) | 1,960 | 1 | 1,212 | 2 | 4 | 0 | 0 | 0 |
| Platform: static shell (blp and others) | 27,211 | 387 | 27,515 | 115 | 91 | 50 | 0 | 4 |
| **SLR total** | **307,737** | **22,666** | **102,920** | **5,026** | **9,687** | **114** | **15** | **175** |
