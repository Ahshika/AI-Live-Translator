"""Setup check: everything Universal Mode needs, with a plain-language fix for each problem."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from app.core.config import Settings



@dataclass
class Check:
    id: str
    ok: bool
    required: bool
    title_ar: str
    detail: str = ""
    fix_ar: str = ""


def run_checks(settings: Settings) -> list[Check]:
    checks: list[Check] = []

    # 1) microphone
    try:
        from app.services.audio.devices import default_input_device, find_device

        mic = find_device(settings.input_device) if settings.input_device else default_input_device()
        checks.append(Check("microphone", mic is not None, True, "الميكروفون",
                            mic.name if mic else "لا يوجد ميكروفون",
                            "" if mic else "وصّل ميكروفونًا أو سماعة رأس بميكروفون."))
    except Exception as exc:
        checks.append(Check("microphone", False, True, "الميكروفون", str(exc), "اختر ميكروفونًا من الإعدادات."))

    # 2) virtual microphone (VB-CABLE)
    from app.services.audio.virtual_mic import VB_CABLE_URL, find_virtual_cable

    cable = find_virtual_cable()
    checks.append(Check(
        "virtual_mic", cable is not None, True, "الميكروفون الوهمي (لإرسال الترجمة للاجتماع)",
        f"{cable.playback.name} → اختر «{cable.mic_name}» كميكروفون في برنامج الاجتماع" if cable else "غير مثبت",
        "" if cable else ("ثبّت برنامج VB-CABLE المجاني:\n"
                          f"١. حمّله من: {VB_CABLE_URL}\n"
                          "٢. فك ضغط الملف.\n"
                          "٣. اضغط بالزر الأيمن على VBCABLE_Setup_x64 واختر «تشغيل كمسؤول» ثم Install.\n"
                          "٤. أعد تشغيل الجهاز.")))

    # 3) meeting audio capture helper
    from app.services.audio import process_capture as pc

    try:
        helper = pc.helper_path()
        ok = pc.supported()
        checks.append(Check("meeting_capture", ok, True, "التقاط صوت الاجتماع", str(helper.name),
                            "" if ok else "يحتاج Windows 10 (إصدار 2004 أو أحدث) أو Windows 11."))
    except pc.MeetingAudioUnavailableError as exc:
        checks.append(Check("meeting_capture", False, True, "التقاط صوت الاجتماع", str(exc),
                            "أعد تثبيت البرنامج (أداة الالتقاط ناقصة)."))
    apps = pc.running_meeting_apps() if sys.platform == "win32" else []
    checks.append(Check("meeting_apps", True, False, "برامج الاجتماعات المفتوحة الآن",
                        "، ".join(apps) if apps else "لا يوجد — افتح زووم أو تيمز أو المتصفح"))

    # 4) GPU
    from app.utils.cuda import cuda_available

    gpu = cuda_available()
    checks.append(Check("gpu", gpu, False, "كرت الشاشة (NVIDIA)",
                        "متاح — ترجمة سريعة" if gpu else "غير متاح — سيعمل على المعالج وسيكون أبطأ",
                        "" if gpu else "حدّث تعريف NVIDIA، أو استخدم نموذج «small» في الإعدادات."))

    # 5) models
    from app.core.config import models_dir

    whisper = models_dir() / f"faster-whisper-{settings.stt_model}" / "model.bin"
    nllb = models_dir() / settings.translation_model / "model.bin"
    for cid, path, title in (("stt_model", whisper, "نموذج فهم الكلام"), ("mt_model", nllb, "نموذج الترجمة")):
        checks.append(Check(cid, path.exists(), True, title, path.parent.name,
                            "" if path.exists() else "شغّل «تنزيل النماذج» من شاشة الإعداد."))
    return checks


def summary(checks: list[Check]) -> dict:
    return {"ready": all(c.ok for c in checks if c.required), "checks": [asdict(c) for c in checks]}
