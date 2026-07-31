# 鍦烘櫙鏁版嵁闆嗕笌浠诲姟闆嗙敓鎴愬懡浠ゆ寚鍗?
鏈枃鎸夋墽琛岄『搴忔眹鎬绘湰椤圭洰浠庢暟鎹暣鐞嗐€佸璞?catalog 鏋勫缓銆丵wen 鎴块棿鎺ㄨ崘銆佹鐜囬噰鏍枫€佹壙杞介潰鎻愬彇銆佹壒閲?layout 鐢熸垚鍒颁换鍔￠泦鐢熸垚鐨勫父鐢ㄥ懡浠ゃ€傛墍鏈夊懡浠ら兘閫氳繃 `log_filter.py` 鍖呰９锛屼究浜庡帇缂?Habitat/OpenGL 鐨勯珮棰戞棩蹇椼€?
榛樿 Qwen 杩炴帴鏂瑰紡涓?SSH 瀵嗙爜鐧诲綍锛?
- host: `7.216.187.6`
- ssh port: `30180`
- user: `root`
- password: `666666`
- remote vLLM API: `127.0.0.1:8000`

## 1. 鏁寸悊椤圭洰鏁版嵁鐩綍

鍏堥瑙堜細绉诲姩鍝簺鐩綍锛?
```bash
python core/log_filter.py --run "python core/prepare_project_structure.py --dry-run"
```

纭鏃犺鍚庢墽琛岃縼绉汇€傝縼绉诲悗鐨勪富瑕佺洰褰曚负锛?
- `data/scenes/hm3d`
- `data/object_datasets/ycb-v1.2`
- `data/object_datasets/hssd-hab-v0.2.3`
- `data/object_images/legacy`
- `data/object_catalog`
- `data/archives`

```bash
python core/log_filter.py --run "python core/prepare_project_structure.py"
```

妫€鏌ユ暟鎹洰褰曪細

```bash
python core/log_filter.py --run "python -c \"from core.project_paths import resolve_hm3d_root, default_object_config_dirs_str; print(resolve_hm3d_root()); print(default_object_config_dirs_str())\""
```

## 2. 鏋勫缓瀵硅薄 Catalog

鏋勫缓缁熶竴瀵硅薄琛ㄣ€俵egacy 瀵硅薄浣跨敤鍥剧墖锛沋CB/HSSD 娌℃湁鍥剧墖鏃朵娇鐢?`semantic_text`銆?
```bash
python core/log_filter.py --run "python core/build_object_catalog.py --datasets legacy,ycb,hssd --write-missing"
```

濡傛灉鍙兂鍏堟鏌ュ璞℃暟閲忥紝涓嶅啓鏂囦欢锛?
```bash
python core/log_filter.py --run "python core/build_object_catalog.py --datasets ycb,hssd --dry-run"
```

濡傛灉 HSSD 瀵硅薄缂哄皯璇箟鎻忚堪锛屽彲鍏堝鍑虹己澶卞垪琛細

```bash
python core/log_filter.py --run "python core/build_object_catalog.py --datasets hssd --write-missing --missing-output data/object_catalog/missing_semantic_text.csv"
```

## 3. 涓?HSSD 琛ュ厖 Semantic Text

HSSD 娌℃湁瀵硅薄鍥剧墖涓旈儴鍒嗛厤缃病鏈夋弿杩版€ц涔夋枃鏈椂锛屽彲鐢ㄩ厤缃厓鏁版嵁鎵归噺璇锋眰 Qwen 鐢熸垚绠€鐭弿杩般€傚厛灏忔壒閲忔祴璇曪細

```bash
python core/log_filter.py --run "python core/generate_object_semantic_text.py --limit 20 --ssh-password 666666"
```

纭杈撳嚭鍚堢悊鍚庣户缁敓鎴愭洿澶氭潯鐩細

```bash
python core/log_filter.py --run "python core/generate_object_semantic_text.py --limit 200 --ssh-password 666666"
```

鐢熸垚鍚庨噸寤?catalog锛岃鏂板鎻忚堪杩涘叆 `object_catalog.json`锛?
```bash
python core/log_filter.py --run "python core/build_object_catalog.py --datasets legacy,ycb,hssd --write-missing"
```

濡傛灉浠嶇己灏戞弿杩帮紝鎺ㄨ崘鏂规鏄細鍏堢敤 Habitat-Sim 绂诲睆娓叉煋瀵硅薄棰勮鍥惧埌 `data/object_previews/hssd`锛屽啀鎶娾€滈瑙堝浘 + config 鍏冩暟鎹€濅竴璧峰彂缁?Qwen 鐢熸垚鏇村彲闈犵殑 `semantic_text`锛涜嫢杩滅涓嶅彲鐢紝鍒欎汉宸ョ紪杈?`data/object_catalog/object_text_overrides.json` 瑕嗙洊鍏抽敭瀵硅薄銆?
## 4. 瀵煎嚭 Scene Info

鍗曞満鏅鍑猴細

```bash
python core/log_filter.py --run "python core/export_scene_info.py --scene 00808-y9hTuugGdiq --data-dir data/scenes/hm3d --output-dir results/scene_info/00808-y9hTuugGdiq"
```

鎵归噺瀵煎嚭鎵€鏈夊彲鐢ㄥ満鏅細

```bash
python core/log_filter.py --run "python core/export_scene_info.py --all --data-dir data/scenes/hm3d --output-dir results/scene_info"
```

## 5. 鐢熸垚瀵硅薄鍒版埧闂寸殑鎺ㄨ崘

legacy 鍥剧墖瀵硅薄锛?
```bash
python core/log_filter.py --run "python core/query_rooms_for_objects.py --scene 00808-y9hTuugGdiq --object-datasets legacy --images-dir data/object_images/legacy --output-dir results/scene_info --ssh-password 666666"
```

YCB text-only 瀵硅薄锛?
```bash
python core/log_filter.py --run "python core/query_rooms_for_objects.py --scene 00808-y9hTuugGdiq --object-datasets ycb --object-catalog data/object_catalog/object_catalog.json --output-dir results/scene_info --ssh-password 666666"
```

HSSD text-only 灏忔壒閲忔祴璇曪細

```bash
python core/log_filter.py --run "python core/query_rooms_for_objects.py --scene 00808-y9hTuugGdiq --object-datasets hssd --object-catalog data/object_catalog/object_catalog.json --limit-objects 20 --output-dir results/scene_info --ssh-password 666666"
```

娣峰悎瀵硅薄闆嗭細

```bash
python core/log_filter.py --run "python core/query_rooms_for_objects.py --scene 00808-y9hTuugGdiq --object-datasets legacy,ycb,hssd --object-catalog data/object_catalog/object_catalog.json --images-dir data/object_images/legacy --limit-objects 50 --output-dir results/scene_info --ssh-password 666666"
```

## 6. 鐢熸垚鎴栧鐢ㄦ鐜囧垎甯?
浠呴噰鏍风敓鎴?layout 鑽夌锛岀己澶辨鐜囨椂鑷姩鏍规嵁鎴块棿鎺ㄨ崘鐢熸垚锛?
```bash
python core/log_filter.py --run "python core/sample_and_place_objects.py --scene 00808-y9hTuugGdiq --mode generate --object-datasets legacy,ycb,hssd --object-catalog data/object_catalog/object_catalog.json --images-dir data/object_images/legacy --limit-objects 50 --rooms-info-dir results/scene_info --probabilities-dir results/probabilities --layouts-dir results/layouts"
```

鍚庣画澶嶇敤姒傜巼鍒嗗竷锛屽彧閲嶆柊 sample锛?
```bash
python core/log_filter.py --run "python core/sample_and_place_objects.py --scene 00808-y9hTuugGdiq --mode load --object-datasets legacy,ycb,hssd --object-catalog data/object_catalog/object_catalog.json --images-dir data/object_images/legacy --limit-objects 50 --rooms-info-dir results/scene_info --probabilities-dir results/probabilities --layouts-dir results/layouts"
```

## 7. 鎻愬彇鍙斁缃壙杞介潰

榛樿浣跨敤 Qwen 杈呭姪鎺掑簭鎵胯浇闈細

```bash
python core/log_filter.py --run "python core/query_room_receptacle_objects.py --scene 00808-y9hTuugGdiq --data-dir data/scenes/hm3d --scene-info-path results/scene_info/00808-y9hTuugGdiq/00808-y9hTuugGdiq_scene_info.json --output results/receptacle_queries/00808-y9hTuugGdiq/00808-y9hTuugGdiq_receptacle_surfaces_all_rooms.json --ssh-password 666666"
```

鏃犺繙绔?LLM 鐨勫惎鍙戝紡妯″紡锛?
```bash
python core/log_filter.py --run "python core/query_room_receptacle_objects.py --scene 00808-y9hTuugGdiq --data-dir data/scenes/hm3d --scene-info-path results/scene_info/00808-y9hTuugGdiq/00808-y9hTuugGdiq_scene_info.json --output results/receptacle_queries/00808-y9hTuugGdiq/00808-y9hTuugGdiq_receptacle_surfaces_all_rooms.json --disable-llm"
```

## 8. 鍗曟 Assignment + Final Layout

鑷姩鐢熸垚 surfaces銆侀噰鏍枫€佸垎閰嶅苟鏀剧疆锛?
```bash
python core/log_filter.py --run "python core/assign_objects_to_receptacle_instances.py --scene 00808-y9hTuugGdiq --object-datasets legacy,ycb,hssd --object-catalog data/object_catalog/object_catalog.json --images-dir data/object_images/legacy --limit-objects 50 --ssh-password 666666"
```

浣跨敤宸叉湁 surfaces锛屽惎鍙戝紡鍒嗛厤锛?
```bash
python core/log_filter.py --run "python core/assign_objects_to_receptacle_instances.py --scene 00808-y9hTuugGdiq --surfaces-json results/receptacle_queries/00808-y9hTuugGdiq/00808-y9hTuugGdiq_receptacle_surfaces_all_rooms.json --object-datasets legacy,ycb,hssd --object-catalog data/object_catalog/object_catalog.json --images-dir data/object_images/legacy --limit-objects 50 --disable-llm"
```

## 9. 鎵归噺鐢熸垚澶氫釜 Layout

鍚屼竴鍦烘櫙鐢熸垚 10 涓渶缁?layout锛屽鐢ㄥ凡鏈夋鐜囧拰 surfaces锛?
```bash
python core/log_filter.py --run "python core/batch_generate_layouts.py --scene 00808-y9hTuugGdiq --num-layouts 10 --object-datasets legacy,ycb,hssd --object-catalog data/object_catalog/object_catalog.json --images-dir data/object_images/legacy --limit-objects 50 --ssh-password 666666"
```

鏃犺繙绔?LLM 鐨勫揩閫?smoke test锛?
```bash
python core/log_filter.py --run "python core/batch_generate_layouts.py --scene 00808-y9hTuugGdiq --num-layouts 2 --object-datasets ycb --object-catalog data/object_catalog/object_catalog.json --disable-assignment-llm --disable-surface-llm"
```

璁″垝妯″紡锛岃鍙栧鍦烘櫙鍒楄〃锛?
```bash
python core/log_filter.py --run "python core/batch_generate_layouts.py --plan-json scenes_plan.json --num-layouts 5 --object-datasets legacy,ycb,hssd --object-catalog data/object_catalog/object_catalog.json --images-dir data/object_images/legacy --limit-objects 50 --ssh-password 666666"
```

璁″垝鏂囦欢绀轰緥锛?
```json
{
  "num_layouts": 5,
  "base_seed": 42,
  "object_datasets": "legacy,ycb,hssd",
  "limit_objects": 50,
  "scenes": [
    "00808-y9hTuugGdiq",
    {"scene": "00800-TEEsavR23oF", "num_layouts": 3, "base_seed": 100}
  ]
}
```

## 10. 鍙鍖栨鏌ュ拰鎵嬪姩淇

鎵撳紑鍗曚釜 layout锛?
```bash
python core/log_filter.py --run "python core/visualize_placed_layout.py results/layouts/00808-y9hTuugGdiq/00808-y9hTuugGdiq_assigned_instance_layout.json --scene 00808-y9hTuugGdiq"
```

鎵撳紑 batch 涓竴涓?layout锛屽苟鐢?`[` / `]` 鍒囨崲鍚岀洰褰曞叾浠?layout锛?
```bash
python core/log_filter.py --run "python core/visualize_placed_layout.py results/layouts/00808-y9hTuugGdiq/batch_20260101_120000/layout_000_seed_42.json --scene 00808-y9hTuugGdiq"
```

鎵嬪姩璋冭瘯楂樺害锛岄粯璁ゅ彲瑙嗗寲鍔犺浇鏃朵細缁欐墍鏈夌墿浣撳簲鐢?`--initial-y-offset 2.5`锛?
```bash
python core/log_filter.py --run "python core/visualize_placed_layout.py results/layouts/00808-y9hTuugGdiq/batch_20260101_120000/layout_000_seed_42.json --scene 00808-y9hTuugGdiq --debug-offset --offset-step 0.02"
```

涓ユ牸澶嶇幇鍘熷 layout锛屼笉鍔犻粯璁?Y 鍋忕Щ锛?
```bash
python core/log_filter.py --run "python core/visualize_placed_layout.py results/layouts/00808-y9hTuugGdiq/batch_20260101_120000/layout_000_seed_42.json --scene 00808-y9hTuugGdiq --initial-y-offset 0"
```

浣跨敤 `test_layout.py` 鎵嬪姩缂栬緫锛?
```bash
python core/log_filter.py --run "python core/test_layout.py 00808-y9hTuugGdiq --layout scene_objects.json --ui-lang zh"
```

## 11. 鐢熸垚浠诲姟闆?
濡傛灉宸叉湁鏈€缁?layout锛屽彲浠ヨ皟鐢ㄥ綋鍓嶄换鍔＄紪鎺掕剼鏈敓鎴?benchmark 浠诲姟闆嗭細

```bash
python core/log_filter.py --run "python core/orchestrate_sd_ovon_complete.py --scene 00808-y9hTuugGdiq --layout results/layouts/00808-y9hTuugGdiq/00808-y9hTuugGdiq_assigned_instance_layout.json"
```

濡傛灉闇€瑕佸厛鐢熸垚瑙傛祴鏁版嵁锛?
```bash
python core/log_filter.py --run "python core/observation_generator.py --scene 00808-y9hTuugGdiq --layout results/layouts/00808-y9hTuugGdiq/00808-y9hTuugGdiq_assigned_instance_layout.json"
```

## 12. 杈撳嚭缁撴瀯閫熻

鏈€缁堜富瑕佷骇鐗╋細

- `results/scene_info/<scene>/<scene>_scene_info.json`锛氬満鏅涔夈€佹埧闂村拰瀹炰緥鏄庣粏銆?- `results/scene_info/<scene>/<object>_rooms.json`锛氭瘡涓璞＄殑鍊欓€夋埧闂存帹鑽愩€?- `results/probabilities/<scene>/<object>_probs.json`锛氬璞″湪鍊欓€夋埧闂翠笂鐨勯噰鏍锋鐜囥€?- `results/receptacle_queries/<scene>/<scene>_receptacle_surfaces_all_rooms.json`锛氭瘡涓埧闂村彲鏀剧疆鎵胯浇闈㈢殑鍊欓€夊疄渚嬪拰琛ㄩ潰鐐逛簯寮曠敤銆?- `results/layouts/<scene>/batch_<time>/layout_<idx>_seed_<seed>.json`锛氭渶缁堝竷灞€銆?- `results/layouts/<scene>/batch_<time>/manifest.json`锛氭壒閲忕敓鎴愭憳瑕併€佸け璐ュ師鍥犲拰澶嶇敤璺緞銆?- `benchmark/...`锛氭牴鎹竷灞€鍜岃娴嬬敓鎴愮殑浠诲姟闆嗐€乪pisode 鎴栬瘎娴嬩骇鐗┿€?
