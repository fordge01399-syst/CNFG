# CNFG-Agent Closed-Loop Project

هذا المشروع يعيد بناء CNFG على هيئة **وكيل مغلق الحلقة**. لا يدخل `optimal_plan` أو `composition` إلى النموذج. في كل خطوة تستدعي البيئة `observe()`، يقرأ الوكيل الحالة والهدف، يختار فعلًا، ثم تستدعي البيئة `step(action)` وتنتج حالة جديدة.

## البيئة والتنصيب

التنفيذ المرجعي استخدم Python 3.12.3 وPyTorch 2.14.0+cpu وNumPy وMatplotlib على CPU دون CUDA.

```bash
python3 -m pip install -r requirements.txt
```

## اختبار الكود

```bash
python3 -m pytest -q tests
```

## توليد البيانات

```bash
python3 scripts/generate_dataset.py \
  --out data --seed 2026 \
  --train 12000 --valid 2000 --test 500 --final 1000
```

ينتج `train.jsonl` و`validation.jsonl` وsplits التعميم والاختبار المستقل. كل سجل يحتوي task metadata وfeatures والفعل الخبير، لكن النموذج يستخدم features فقط.

## تدقيق التسريب

```bash
python3 scripts/leakage.py --data data --out results/leakage_audit.json
```

يجب تنفيذ التدقيق قبل اختيار checkpoint أو قراءة نتائج الاختبار النهائي.

## التدريب متعدد البذور

لإعادة تشغيل الجولة المحسنة v4، استخدم البيانات الجديدة والتدريب التسلسلي الذي يحافظ على الذاكرة عبر حلقات كاملة:

```bash
python3 scripts/generate_dataset.py --out data_v4 --seed 4040 --train 12000 --valid 2000 --test 500 --final 1000
python3 scripts/leakage.py --data data_v4 --out results/leakage_v4.json
for seed in 42 43 44 45 46; do
  python3 scripts/train_sequence.py --data data_v4 --out checkpoints_v4 \
    --model cnfg --seed "$seed" --epochs 8 \
    --batch-episodes 128 --dim 96
done
```


```bash
for seed in 42 43 44 45 46; do
  for model in cnfg gru mlp; do
    python3 scripts/train.py --data data --out checkpoints \
      --model "$model" --seed "$seed" --epochs 10 \
      --batch 1024 --dim 64
  done
done
```

اختيار checkpoint يتم من validation فقط.

## تشغيل Agent مغلق الحلقة

مثال v4:

```bash
python3 scripts/evaluate.py \
  --checkpoint checkpoints_v4/cnfg_seed42.pt \
  --model cnfg --data data_v4 --out results/v4_run \
  --split partial_observation_test --max-tasks 200 --dim 96
```


```bash
python3 scripts/evaluate.py \
  --checkpoint checkpoints/cnfg_seed42.pt \
  --model cnfg --data data --out results/run \
  --split iid_test --max-tasks 200
```

الـsplits المتاحة تشمل `iid_test` و`novel_layout_test` و`novel_task_test` و`long_horizon_test` و`partial_observation_test` و`final_independent_test`.

لاختبار الأفعال غير المقنّعة:

```bash
python3 scripts/evaluate.py \
  --checkpoint checkpoints/cnfg_seed42.pt \
  --model cnfg --data data --out results/stress \
  --split iid_test --max-tasks 200 --no-action-mask
```

ولاختبار فشل البيئة:

```bash
python3 scripts/evaluate.py \
  --checkpoint checkpoints/cnfg_seed42.pt \
  --model cnfg --data data --out results/recovery \
  --split iid_test --max-tasks 200 --fail-rate 0.1
```

## اختبار الذاكرة

```bash
python3 scripts/memory_test.py \
  --checkpoint checkpoints/cnfg_seed42.pt \
  --data data --gap 4 --max-tasks 200
```

يُعرض الوكيل كامل observation لعدد `gap` من الخطوات ثم تُخفى مواضع المفتاح والباب.

## Baselines

```bash
python3 scripts/evaluate_baselines.py \
  --data data --split iid_test --agent random --max-tasks 200
python3 scripts/evaluate_baselines.py \
  --data data --split iid_test --agent greedy --max-tasks 200
```

الـGRU والـMLP يتدربان ويقيّمان عبر `train.py` و`evaluate.py` بنفس واجهة features.

## المخرجات

التقرير النهائي هو `FINAL_REPORT.md`. النتائج الخام في `results/`، والـcheckpoints في `checkpoints/`، والرسوم في `plots/`، والسجلات في `logs/`. ملفات `results/pre_feature_fix/` محفوظة لأغراض التدقيق لأنها تمثل محاولة أولى قبل إصلاح تمثيل key/door؛ لا تُخلط مع النتائج المصححة.

## تفسير النتائج

لا ينبغي تفسير Action Accuracy كـFull Task Success. المقياس الأهم هو نجاح المهمة كاملة عبر rollout. التقرير الحالي يخلص إلى أن CNFG-Agent قابل للتشغيل مغلق الحلقة، لكنه أضعف من GRU وMLP في هذه التجربة، وفشل في الملاحظات الجزئية واختبار فجوة الذاكرة. هذه النتيجة مقصودة ومُسجلة كما هي.


## DAgger long-horizon improvement

لتدريب النسخة التي تتعرض لأخطائها وتعيد التخطيط من الحالة الفعلية:

```bash
for seed in 42 43 44 45 46; do
  python3 scripts/train_dagger.py \
    --data data_v4 \
    --out checkpoints_dagger \
    --init checkpoints_v4/cnfg_seed${seed}.pt \
    --seed "$seed" --epochs 6 \
    --episodes 256 --dim 96 --max-steps 140
done
```

للتقييم:

```bash
python3 scripts/evaluate.py \
  --checkpoint checkpoints_dagger/cnfg_seed42.pt \
  --model cnfg --data data_v4 --out results/dagger_run \
  --split long_horizon_test --max-tasks 200 --dim 96
```

هذه الجولة حسّنت long-horizon وfinal independent، لكنها سببت trade-off مع IID؛ لذلك لا تُعتبر نجاحًا شاملًا قبل تنفيذ mixed DAgger وتقييمه على جميع splits.
