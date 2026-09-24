# CNFG_PRETRAINED_WEIGHT_TRANSFER_REPORT

## 1. Research Question

هل يمكن للمعرفة المسبقة من Transformer لغوي كبير نسبيًا أن تنتقل إلى CNFG-Agent، وهل تحسن هذه المعرفة التخطيط المغلق الحلقة، خصوصًا في المهام طويلة الأفق والاختبار المستقل؟ لا تُعامل هذه الجولة كدليل على تفوق CNFG؛ بل كاختبار مباشر لفرضية نقل التمثيل.

## 2. Hypotheses

الفرضية الأولى أن تمثيل Qwen المسبق يمكن أن يحسن ترميز الحالة والهدف مقارنة برأس صغير يبدأ عشوائيًا. الفرضية الثانية أن نقل تمثيل Qwen إلى وحدة recurrent/compositional صغيرة قد يحسن الثبات عبر الخطوات. الفرضية الثالثة أن أي تحسن يجب أن يظهر في closed-loop execution، لا في دقة تصنيف انتقالات منفردة فقط.

## 3. Experimental Setup

استخدمت الجولة نموذج `Qwen/Qwen2.5-0.5B-Instruct` الرسمي من Hugging Face، بتمثيل مخفي حجمه 896 وبنحو 494M معلمة. جرى تنزيله إلى `pretrained/qwen25_0.5b/` وتسجيل metadata. استُخدمت بيانات `data_v4`، لكن هذه الجولة التجريبية الأولية استخدمت 256 انتقال تدريب و64 انتقال تحقق بسبب كلفة تشغيل Qwen CPU في البيئة الحالية؛ لذلك لا يجوز تعميمها على الأداء النهائي.

النص المرسل إلى Qwen صيغ من `state`, `goal`, `observations`, `visible entities`, `walls`, `distractors`, و`valid_actions`. حُذفت صراحةً `expert_action`, `optimal_steps`, وأي تسلسل أفعال. جميع النتائج في هذه الجولة اختيارية/استكشافية، ولم يُستخدم أي اختبار نهائي لاختيار checkpoint.

## 4. Architecture

تتكون تجربة Frozen-Qwen من Qwen مجمد بالكامل مع رأس action صغير. تجربة Fine-tuned-Qwen تجمد كل Qwen عدا آخر transformer block وتدرّبه مع رأس action. تجربة CNFG-Qwen transfer تستخدم Qwen كمستخرج تمثيل مجمد ثم تنقل هذا التمثيل إلى وحدة `GRUCell` مع رأس أفعال؛ وهي نقل تمثيل إلى مسار ذاكرة/تركيب، وليست ادعاءً بأن أوزان Qwen وCNFG متطابقة بنيويًا.

في CNFG-Agent الأساسي، العقد هي تمثيلات الحالة والهدف والكيانات، والعلاقات هي الفروق المكانية والسمات typed pair features، ثم تمرر إلى الذاكرة recurrent وتُسجل الأفعال مع قناع صلاحية منفصل. هذه الجولة لا تدخل خطة الخبير إلى النموذج.

## 5. Pretrained Backbone Method

استُخدم Qwen كـencoder للنص المنظم. يتم average-pooling على آخر hidden states باستخدام attention mask. في Frozen-Qwen لا تتغير أوزان Qwen. في Fine-tuned-Qwen تُفتح آخر طبقة فقط مع رأس الأفعال. في Transfer تُحفظ تمثيلات Qwen ثم يُدرّب عليها projection وGRUCell ورأس الأفعال.

## 6. Weight Transfer Method

النقل المطبق هو **representation transfer**: hidden states من Qwen تُستخدم كمدخل لمكوّن transfer/recurrent في CNFG-side policy. لم تُنسخ أوزان attention إلى CNFG لأن البنى غير متوافقة، ولم يُسمَّ ذلك نسخًا مباشرًا للأوزان. هذا التمييز محفوظ في الكود والتقرير.

## 7. Baselines

توجد نتائج CNFG وGRU وMLP وRandom وGreedy في `results/` من الجولات السابقة، وتوجد نتائج DAgger CNFG في `results/dagger_summary.json`. أضيفت في هذه الجولة Frozen-Qwen وFine-tuned-Qwen وCNFG-Qwen transfer. لم تُنفذ بعد مقارنة Transformer/GNN جديدة مطابقة المعلمات ضمن نفس بروتوكول Qwen بسبب كلفة تشغيل Qwen على CPU، ولذلك لا أدعي اكتمال مصفوفة baselines المطلوبة.

## 8. DAgger Protocol

في الجولة السابقة `v4.1.0` استُخدم DAgger-style collection: يُسمح للوكيل باتخاذ أفعال أثناء rollout، ثم تُعاد تسمية الحالات التي وصل إليها بالفعل الأمثل من الحالة الفعلية الحالية. حقق ذلك Long Horizon بنسبة **86.4% ± 13.7%** وFinal Independent بنسبة **50.8% ± 11.7%**، مقابل **30.5% ± 8.3%** و**31.1% ± 9.9%** في v4. لكنه خفض IID إلى **58.6% ± 9.7%**، لذلك ما زال يحتاج Mixed DAgger.

## 9. Parameter Matching

Qwen نفسه نحو 494M معلمة، بينما رأس Frozen/Fine-tuned صغير مقارنة به، وTransfer head نحو 0.3M تقريبًا. هذه ليست مقارنة parameter-matched مع CNFG. سُجلت أعداد المعلمات في checkpoint metadata. يجب تنفيذ مقارنة مستقلة بحجم Transformer/GNN/GRU مقارب قبل أي استنتاج عن الكفاءة.

## 10. Results

| Model | Validation action accuracy | Data regime |
|---|---:|---:|
| Frozen-Qwen | 14.1% | 256 train / 64 validation transitions |
| Fine-tuned-Qwen, last block | 14.1% | 256 train / 64 validation transitions |
| CNFG-Qwen transfer | 21.9% | 256 train / 64 validation transitions |

نتيجة transfer أعلى من Frozen/Fine-tuned في هذا pilot، لكنها نتيجة صغيرة ومحدودة ولا تمثل closed-loop task success. كما أن القيم منخفضة، ما يدل على أن Qwen النصي لم يتعلم سياسة الأفعال من هذا الحجم الصغير في جولة واحدة.

## 11. Statistical Analysis

لم تُجرَ دلالة إحصائية على pilot Qwen لأنه seed واحد وعينة صغيرة. النتائج القابلة للمقارنة إحصائيًا في DAgger استخدمت خمس بذور، والانحرافات المعيارية population SD كما هو موثق في ملفات JSON. لا ينبغي دمج pilot Qwen مع DAgger في متوسط واحد.

## 12. Ablations

المنفذ حاليًا هو Frozen مقابل Fine-tuned-last-block مقابل CNFG-side transfer. ما لم يُنفذ بعد: CNFG - Pair Memory، CNFG - Recurrence، CNFG - Composition، CNFG - Goal Encoder، CNFG - Relation Module، CNFG - Memory، وCNFG صغير/كبير ضمن مسار Qwen نفسه. هذه البنود تبقى قائمة صراحةً بدل ملء جدولها بنتائج غير موجودة.

## 13. Failure Analysis

الفشل الرئيسي في Qwen pilot هو صغر البيانات وكلفة encoder العالية على CPU. أُجري smoke test مغلق الحلقة على 5 مهام long-horizon باستخدام Frozen-Qwen؛ بدون قناع الصلاحية كان `full_task_success=0.0` و`invalid_action_rate=98.14%`، مع وصول جميع الحلقات إلى حد 140 خطوة دون إنهاء. أُوقف تشغيل النسخة المقنّعة بعد تسجيل هذا الفشل لأن استدلال Qwen 0.5B على CPU كان بطيئًا جدًا، ولذلك لا تُسجل لها نتيجة غير مكتملة. انخفاض دقة Fine-tuned عن المتوقع لا يثبت أن Qwen غير مفيد؛ بل يثبت أن آخر طبقة وتدريبًا واحدًا على 256 انتقالًا غير كافيين. كما أن transfer الحالي ينقل تمثيلًا لا خطة، ولذلك يحتاج إلى sequence training وrollout exposure.

## 14. Limitations

هذه الجولة ليست برهانًا على نجاح pretrained transfer. لم تُستخدم 500,000–2,000,000 حلقة لأن تشغيل Qwen 0.5B على CPU غير عملي ضمن البيئة الحالية. لم تُنفذ بعد كل ablations أو Transformer/GNN parameter matching، ولم يُستكمل closed-loop evaluation واسع لـQwen؛ المتاح فقط هو smoke test الفاشل الموثق أعلاه. يجب تنفيذ ذلك قبل إصدار ادعاء نهائي.

## 15. Reproducibility

الأوامر الأساسية:

```bash
python3 scripts/probe_qwen.py
python3 scripts/qwen_transfer.py --data data_v4 --out experiments/pretrained_backbone --train-rows 256 --valid-rows 64 --epochs 1 --seed 42
python3 scripts/evaluate_qwen.py --checkpoint experiments/pretrained_backbone/frozen_qwen_seed42.pt --split long_horizon_test --tasks 5 --seed 42
python3 -m pytest -q tests
```

معرف النموذج هو `Qwen/Qwen2.5-0.5B-Instruct`. البيانات هي `data_v4`، والـcommit الأساسي السابق هو `1bc13e0` قبل إضافة هذا المسار. يجب تسجيل commit الجديد عند رفع التقرير. جميع ملفات metadata والـcheckpoints في `pretrained/` و`experiments/pretrained_backbone/`.

## 16. Conclusions

أثبتت الجولة أن مسار Qwen قابل للتشغيل، وأن نقل التمثيل إلى رأس recurrent أعطى 21.9% validation action accuracy مقابل 14.1% لـFrozen/Fine-tuned في pilot صغير. لكنها لم تثبت بعد تحسن closed-loop أو Final Independent. أقوى دليل حالي على رفع Long Horizon ما زال DAgger CNFG، الذي وصل إلى 86.4% في Long Horizon و50.8% في Final Independent. الخطوة العلمية الصحيحة التالية هي تشغيل Mixed DAgger، وتوسيع Qwen data regime، وإجراء closed-loop evaluation مستقل مع Transformer/GNN parameter matching.

## References

[1]: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct "Qwen2.5-0.5B-Instruct model card"
[2]: https://github.com/QwenLM/Qwen2.5 "Qwen2.5 official repository"
