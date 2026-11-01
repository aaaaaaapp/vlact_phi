# Pure PI / tanh 八任务共同执行 800 seed

本清单为原始测评加补测后两模型均实际执行的 seed；每任务 100 个。原始预筛清单保留不变。

| 任务 | 共同执行 | Pure PI 成功 | tanh 成功 | tanh 独赢 | Pure PI 独赢 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `click_alarmclock` | 100 | 95 | 71 | 5 | 29 |
| `beat_block_hammer` | 100 | 88 | 78 | 8 | 18 |
| `open_microwave` | 100 | 99 | 83 | 1 | 17 |
| `shake_bottle` | 100 | 100 | 100 | 0 | 0 |
| `move_playingcard_away` | 100 | 100 | 77 | 0 | 23 |
| `pick_dual_bottles` | 100 | 91 | 62 | 5 | 34 |
| `handover_block` | 100 | 96 | 81 | 2 | 17 |
| `place_object_basket` | 100 | 90 | 79 | 6 | 17 |
| **合计** | **800** | **759** | **631** | **27** | **155** |

机器清单：`docs/robotwin_tanh02_pure_full8_executed_800_seeds.json`。
补测原始记录：`logs/tanh02_pure_full8_supplement_20261030/`。
