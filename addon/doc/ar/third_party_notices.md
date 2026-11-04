# إشعارات المكونات الخارجية

تُرخّص إضافة Offline AI نفسها بموجب رخصة GNU العمومية، الإصدار الثاني؛ راجع النص الأصلي في `COPYING.txt`. هذا الملف ترجمة توضيحية لإشعارات المكونات، ولا يحل محل نصوص تراخيصها الأصلية.

تتضمن الإضافة المكونات الآتية دون تعديل، داخل `addon/globalPlugins/offlineAI/lib`. يحتفظ كل مكوّن بترخيصه الخاص.

* **llama.cpp:** لتشغيل نماذج اللغة والرؤية وتوليد الكلام، بما في ذلك `llama.dll` و`ggml*.dll` و`mtmd.dll` و`llama-tts.exe` و`llama-mtmd-cli.exe`. الترخيص: MIT.
* **llama-cpp-python:** واجهة Python لمحرك llama.cpp. الترخيص: MIT.
* **whisper.cpp:** للتعرف على الكلام. الترخيص: MIT.
* **pywhispercpp:** واجهة Python لمحرك whisper.cpp. الترخيص: MIT.
* **NumPy:** للتعامل مع البيانات الصوتية في الذاكرة. الترخيص: BSD ذو البنود الثلاثة.
* **miniaudio وpyminiaudio:** لفك ترميز MP3 وFLAC وOGG وWAV. التراخيص: الملكية العامة أو MIT No Attribution لمحرك miniaudio، وMIT لواجهة Python.
* **python-sounddevice وPortAudio:** لتسجيل الميكروفون وتشغيل الصوت. الترخيص: MIT.
* **cffi وpycparser:** متطلبات لمكتبات الصوت. الترخيصان: MIT وBSD ذو البنود الثلاثة.
* **pypdfium2 وPDFium:** لتحويل صفحات PDF إلى صور. التراخيص المذكورة للمكونات: Apache-2.0 أو BSD ذو البنود الثلاثة؛ تُراجع ملفات كل مكوّن لتحديد الشروط المطبقة عليه.
* **Jinja2 وMarkupSafe:** لمعالجة قوالب المحادثة المحفوظة في ملفات النماذج. الترخيص: BSD ذو البنود الثلاثة.
* **diskcache:** متطلب لمكتبة llama-cpp-python. الترخيص: Apache-2.0.
* **typing_extensions وpickletools:** لتوفير وظائف توافق من مكتبة Python القياسية. الترخيص: PSF.
* **platformdirs:** متطلب لمكتبة pywhispercpp. الترخيص: MIT.
* **مكتبات تشغيل Microsoft Visual C++ وOpenMP:** مثل `msvcp140` و`vcomp140` و`libomp`، لتشغيل المكونات السابقة. تخضع لشروط إعادة التوزيع الخاصة بـ Microsoft أو لرخصة Apache-2.0 مع استثناء LLVM، بحسب المكوّن.

**النماذج ليست جزءًا من حزمة الإضافة.** ينزّلها المستخدم من ناشريها، وغالبًا عبر HuggingFace. لكل نموذج ترخيص مستقل منشور في صفحته؛ تحقّق منه قبل الاستخدام التجاري.

نظام البناء، بما فيه `sconstruct` و`site_scons`، مستمد من قالب إضافات NVDA ويخضع لرخصة GPL v2.

[الإشعارات الأصلية وروابط المشاريع](https://github.com/riadassoum/OfflineAI/blob/main/THIRD_PARTY_NOTICES.md)
