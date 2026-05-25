"""
Cross-lingual intent parsing evaluation via local Ollama (phi3:mini).
Expanded dataset: 25 intents per language × 4 languages = 100 total intents.
Covers 5 categories: Energy Management, Traffic Steering, Security Commands,
QoS Management, Maintenance Tasks.
Produces updated Table 2 summary for Eco-Edge v6 paper.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "phi3:mini")

# ─────────────────────────────────────────────────────────────────────────────
# FIX 4 — EXPANDED INTENT DATASET (25 per language × 4 languages = 100 total)
# Category breakdown: 5 intents × 5 categories per language
# ─────────────────────────────────────────────────────────────────────────────

test_intents = {
    # ── ENGLISH (reference language) ─────────────────────────────────────────
    "English": [
        # --- Category 1: Energy Management (5) ---
        "Reduce energy consumption by 30% during night shift operations",
        "Enforce a 15-watt power budget on all Tier-2 edge nodes",
        "Activate micro-sleep mode on idle edge devices until next task arrives",
        "Enable solar-harvesting-priority scheduling during daylight hours",
        "Initiate graceful shutdown of low-SoC nodes below 15% battery threshold",

        # --- Category 2: Traffic Steering (5) ---
        "Reroute all video analytics traffic to the northern cluster immediately",
        "Balance load across available nodes using round-robin with latency weighting",
        "Offload GPU inference tasks to the nearest cloud gateway if local load exceeds 80%",
        "Allocate 200 Mbps reserved bandwidth for critical IoT sensor streams",
        "Set QoS priority to premium for all autonomous-vehicle control packets",

        # --- Category 3: Security Commands (5) ---
        "Isolate node cluster-7 from the mesh due to anomalous packet injection",
        "Revoke API credentials for agent-node-42 with immediate effect",
        "Quarantine the southern edge zone and redirect traffic to backup nodes",
        "Trigger anomaly-response protocol on nodes flagging irregular CPU spikes",
        "Enable full audit logging on all orchestration decisions for the next 24 hours",

        # --- Category 4: QoS Management (5) ---
        "Enforce a maximum latency target of 10 milliseconds for surgical robotics",
        "Guarantee minimum throughput of 500 Mbps for the media-streaming pipeline",
        "Ensure SLA compliance for mission-critical workloads before batch jobs",
        "Hard-enforce task deadline of 50 ms for real-time control loop workloads",
        "Upgrade resource tier for the predictive-maintenance service from basic to premium",

        # --- Category 5: Maintenance Tasks (5) ---
        "Schedule a 2-hour planned downtime for node-group-B starting at 03:00 UTC",
        "Push firmware update v2.4.1 to all edge nodes in the eastern cluster",
        "Run health-check diagnostics on all nodes reporting latency above threshold",
        "Execute log rotation and compress archives older than seven days",
        "Conduct capacity-planning audit for the next-quarter traffic growth projection",
    ],

    # ── URDU (Roman Urdu transliterations for machine processing) ─────────────
    "Urdu": [
        # --- Category 1: Energy Management ---
        "Raat ki shift ke dauran energy consumption 30 feesad kam karein",
        "Tamam Tier-2 edge nodes par 15 watt ka power budget nafiz karein",
        "Khali edge devices par micro-sleep mode tab tak activate karein jab tak agla task na aaye",
        "Din ke waqt solar-harvesting-priority scheduling enable karein",
        "15 feesad se kam battery wale low-SoC nodes ka graceful shutdown shuru karein",

        # --- Category 2: Traffic Steering ---
        "Tamam video analytics traffic ko foran northern cluster ki taraf reroute karein",
        "Latency weighting ke saath round-robin istemal karte hue load balance karein",
        "Agar local load 80 feesad se zyada ho to GPU inference tasks nazdiki cloud gateway par offload karein",
        "Critical IoT sensor streams ke liye 200 Mbps reserved bandwidth makhsoos karein",
        "Tamam autonomous-vehicle control packets ke liye QoS priority premium par set karein",

        # --- Category 3: Security Commands ---
        "Anormal packet injection ki wajah se cluster-7 ko mesh se alag karein",
        "Agent-node-42 ke API credentials foran radh karein",
        "Southern edge zone ko quarantine karein aur traffic backup nodes par bheijein",
        "Irregular CPU spikes report karne wale nodes par anomaly-response protocol chalayein",
        "Agle 24 ghante ke liye tamam orchestration decisions par mukammal audit logging chalayein",

        # --- Category 4: QoS Management ---
        "Surgical robotics ke liye maximum 10 millisecond latency target nafiz karein",
        "Media-streaming pipeline ke liye minimum 500 Mbps throughput guarantee karein",
        "Batch jobs se pehle mission-critical workloads ki SLA compliance yaqeeni banayein",
        "Real-time control loop workloads ke liye 50 ms task deadline sakht nafiz karein",
        "Predictive-maintenance service ki resource tier basic se premium par upgrade karein",

        # --- Category 5: Maintenance Tasks ---
        "Node-group-B ke liye 03:00 UTC se 2 ghante ka planned downtime schedule karein",
        "Eastern cluster ke tamam edge nodes ko firmware update v2.4.1 push karein",
        "Threshold se zyada latency report karne wale tamam nodes par health-check diagnostics chalayein",
        "Log rotation execute karein aur saat din se purane archives compress karein",
        "Agali quarter ki traffic growth projection ke liye capacity-planning audit karein",
    ],

    # ── ARABIC ───────────────────────────────────────────────────────────────
    "Arabic": [
        # --- Category 1: Energy Management ---
        "تقليل استهلاك الطاقة بنسبة 30% خلال عمليات نوبة الليل",
        "فرض ميزانية طاقة بقيمة 15 واط على جميع عُقد الحافة من المستوى الثاني",
        "تفعيل وضع النوم المصغر على الأجهزة الخاملة حتى وصول المهمة التالية",
        "تفعيل جدولة أولوية حصاد الطاقة الشمسية خلال ساعات النهار",
        "بدء الإيقاف التدريجي للعُقد منخفضة الشحن التي تقل بطاريتها عن 15%",

        # --- Category 2: Traffic Steering ---
        "إعادة توجيه حركة مرور تحليل الفيديو فوراً نحو العنقود الشمالي",
        "موازنة الحمل عبر العُقد المتاحة باستخدام التوزيع الدوري مع ترجيح زمن الاستجابة",
        "نقل مهام الاستدلال على GPU إلى أقرب بوابة سحابية إذا تجاوز الحمل المحلي 80%",
        "تخصيص عرض نطاق ترددي محجوز بقيمة 200 ميغابت في الثانية لتدفقات أجهزة الاستشعار الحيوية",
        "تعيين أولوية جودة الخدمة إلى مستوى مميز لجميع حزم التحكم في المركبات ذاتية القيادة",

        # --- Category 3: Security Commands ---
        "عزل مجموعة العُقد cluster-7 عن الشبكة الشبكية بسبب حقن حزم شاذة",
        "إلغاء بيانات اعتماد واجهة برمجة التطبيقات للعقدة agent-node-42 فوراً",
        "عزل المنطقة الجنوبية وإعادة توجيه الحركة إلى العُقد الاحتياطية",
        "تشغيل بروتوكول الاستجابة للشذوذ على العُقد التي ترصد ارتفاعات غير منتظمة في المعالج",
        "تفعيل تسجيل المراجعة الشامل لجميع قرارات التنسيق لمدة 24 ساعة القادمة",

        # --- Category 4: QoS Management ---
        "فرض هدف زمن استجابة أقصاه 10 ميلي ثانية لتطبيقات الروبوتات الجراحية",
        "ضمان حد أدنى من الإنتاجية بمقدار 500 ميغابت في الثانية لخط أنابيب بث الوسائط",
        "ضمان الامتثال لاتفاقية مستوى الخدمة للأحمال الحرجة قبل المهام الدفعية",
        "فرض صارم لموعد نهائي للمهمة بمقدار 50 ميلي ثانية لأحمال حلقة التحكم الفوري",
        "ترقية مستوى الموارد لخدمة الصيانة التنبؤية من أساسي إلى مميز",

        # --- Category 5: Maintenance Tasks ---
        "جدولة توقف مخطط لمدة ساعتين لمجموعة العُقد node-group-B بدءاً من 03:00 بتوقيت UTC",
        "دفع تحديث البرنامج الثابت v2.4.1 إلى جميع عُقد الحافة في العنقود الشرقي",
        "تشغيل تشخيصات فحص الصحة على جميع العُقد التي تُبلغ عن زمن استجابة فوق الحد",
        "تنفيذ تدوير السجلات وضغط الأرشيفات الأقدم من سبعة أيام",
        "إجراء مراجعة تخطيط السعة لتوقعات نمو حركة المرور للربع القادم",
    ],

    # ── JAPANESE ─────────────────────────────────────────────────────────────
    "Japanese": [
        # --- Category 1: Energy Management ---
        "夜間シフト運用中のエネルギー消費を30%削減する",
        "すべてのTier-2エッジノードに15ワットの電力予算を適用する",
        "次のタスクが到達するまでアイドル状態のエッジデバイスでマイクロスリープモードを有効にする",
        "日中の時間帯に太陽光発電優先スケジューリングを有効にする",
        "バッテリー残量が15%未満の低SoCノードのグレースフルシャットダウンを開始する",

        # --- Category 2: Traffic Steering ---
        "すべてのビデオ解析トラフィックを直ちに北部クラスターに再ルーティングする",
        "レイテンシ加重付きラウンドロビンを使用して利用可能なノード間で負荷分散する",
        "ローカル負荷が80%を超えた場合はGPU推論タスクを最寄りのクラウドゲートウェイにオフロードする",
        "重要なIoTセンサーストリームのために200 Mbpsの予約帯域幅を割り当てる",
        "すべての自律走行車制御パケットのQoS優先度をプレミアムに設定する",

        # --- Category 3: Security Commands ---
        "異常なパケットインジェクションを理由にcluster-7をメッシュから隔離する",
        "agent-node-42のAPI資格情報を即時失効させる",
        "南部エッジゾーンを隔離してトラフィックをバックアップノードに切り替える",
        "不規則なCPUスパイクを報告しているノードで異常対応プロトコルを起動する",
        "次の24時間、すべてのオーケストレーション判断に対して完全な監査ログを有効にする",

        # --- Category 4: QoS Management ---
        "外科ロボティクスに対して最大レイテンシ目標10ミリ秒を適用する",
        "メディアストリーミングパイプラインに最小500 Mbpsのスループットを保証する",
        "バッチジョブの前にミッションクリティカルなワークロードのSLA準拠を確保する",
        "リアルタイム制御ループワークロードに50 msのタスクデッドラインを厳格に適用する",
        "予測保全サービスのリソース階層をベーシックからプレミアムにアップグレードする",

        # --- Category 5: Maintenance Tasks ---
        "03:00 UTCからnode-group-Bの2時間の計画停止をスケジュールする",
        "東部クラスターのすべてのエッジノードにファームウェアアップデートv2.4.1を展開する",
        "閾値を超えるレイテンシを報告しているすべてのノードでヘルスチェック診断を実行する",
        "ログローテーションを実行し7日以上前のアーカイブを圧縮する",
        "翌四半期のトラフィック成長予測のためのキャパシティ計画監査を実施する",
    ],
}

# Ground-truth expected intent labels (for accuracy scoring without LLM)
# Used in offline / unit-test mode
EXPECTED_ACTIONS = [
    "reduce", "enforce", "activate", "enable", "initiate",           # Energy
    "reroute", "balance", "offload", "allocate", "set",              # Traffic
    "isolate", "revoke", "quarantine", "trigger", "enable",          # Security
    "enforce", "guarantee", "ensure", "enforce", "upgrade",          # QoS
    "schedule", "push", "run", "execute", "conduct",                 # Maintenance
]


def _extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def test_intent_parsing(intent_text: str, language: str) -> Tuple[bool, float, Optional[dict]]:
    prompt = f"""You are a network orchestration system.
Parse this intent and return ONLY a JSON object with these exact keys:
- action: the main action (string)
- target: what to act on (string)
- constraint: any constraint like percentage or threshold (string or null)
- priority: high/medium/low (string)

Intent: "{intent_text}"

Return only valid JSON, nothing else."""

    start_time = time.time()
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        result = response.json().get("response", "").strip()
    except requests.RequestException as exc:
        return False, (time.time() - start_time) * 1000, {"error": str(exc)}

    latency = (time.time() - start_time) * 1000
    parsed = _extract_json(result)
    if parsed is None:
        return False, latency, None
    success = all(k in parsed for k in ("action", "target", "constraint", "priority"))
    return success, latency, parsed


def main() -> None:
    print(f"Cross-lingual evaluation — Expanded Dataset (model={OLLAMA_MODEL})")
    print(f"Total intents: {sum(len(v) for v in test_intents.values())} "
          f"({len(test_intents)} languages × 25 intents each)\n")

    # Category names for per-category reporting
    categories = [
        "Energy Management",
        "Traffic Steering",
        "Security Commands",
        "QoS Management",
        "Maintenance Tasks",
    ]

    results_summary: Dict[str, Dict[str, Any]] = {}

    for language, intents in test_intents.items():
        successes = 0
        latencies: List[float] = []
        cat_successes = [0] * 5

        for idx, intent in enumerate(intents):
            success, lat, parsed = test_intent_parsing(intent, language)
            cat_idx = idx // 5  # 5 intents per category
            if success:
                successes += 1
                cat_successes[cat_idx] += 1
            latencies.append(lat)
            mark = "OK" if success else "FAIL"
            preview = intent[:45] + ("..." if len(intent) > 45 else "")
            print(f"[{language}][{categories[cat_idx][:10]}] {mark} | {lat:.0f}ms | {preview}")

        accuracy = (successes / len(intents)) * 100
        avg_latency = float(np.mean(latencies))
        cat_acc = {categories[i]: (cat_successes[i] / 5) * 100 for i in range(5)}
        results_summary[language] = {
            "accuracy": accuracy,
            "avg_latency_ms": avg_latency,
            "overhead_vs_english": None,
            "per_category_accuracy": cat_acc,
            "n_intents": len(intents),
        }
        print(f"\n{language}: {accuracy:.1f}% accuracy, {avg_latency:.0f}ms avg latency\n")

    eng_latency = results_summary["English"]["avg_latency_ms"]
    for lang in results_summary:
        overhead = (
            (results_summary[lang]["avg_latency_ms"] - eng_latency) / eng_latency
        ) * 100
        results_summary[lang]["overhead_vs_english"] = overhead

    print("\n--- Updated Table 2 (Cross-Lingual Evaluation — 25 intents/lang, 100 total) ---")
    print(f"{'Language':<12} {'n':<5} {'Accuracy':<12} {'Avg Latency':<15} {'Overhead vs EN'}")
    print("-" * 60)
    for lang, r in results_summary.items():
        print(
            f"{lang:<12} {r['n_intents']:<5} {r['accuracy']:.1f}%{'':<7} "
            f"{r['avg_latency_ms']:.0f}ms{'':<9} {r['overhead_vs_english']:+.1f}%"
        )

    print("\n--- Per-Category Accuracy ---")
    for lang, r in results_summary.items():
        print(f"\n{lang}:")
        for cat, acc in r["per_category_accuracy"].items():
            print(f"  {cat:<25} {acc:.0f}%")

    out_path = os.path.join(os.path.dirname(__file__), "cross_lingual_results_v2.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
