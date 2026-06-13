_LANG_NAMES = {
    'es': 'Spanish',
    'en': 'English',
    'fr': 'French',
    'de': 'German',
    'ru': 'Russian',
    'pt': 'Portuguese',
}

def _lang_name(code):
    return _LANG_NAMES.get(code.lower(), code)


_TARGET_LANG_STYLES = {
    'en': 'English at B2 level',
    'es': 'Spanish',
    'ru': 'Russian',
}

# Provider-agnostic JSON Schema for the combined evaluate + summarize call.
# One LLM request per article instead of two — on the free tier the binding
# constraint is requests per day, not tokens, so the always-generated summary
# (discarded for below-threshold articles) is the cheaper side of the trade.
# Each provider wraps this in its own request-format envelope.
EVALUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "expat_impact": {"type": "integer", "minimum": 1, "maximum": 10, "description": "How relevant or impactful the news is for expatriates (1-10)"},
        "event_weight": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Significance or uniqueness of the event (1-10)"},
        "politics": {"type": "integer", "minimum": 0, "maximum": 10, "description": "Non-political/innovation score (0=political, 10=non-political/innovative)"},
        "timeliness": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Time-sensitivity or urgency (1-10)"},
        "practical_utility": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Usefulness for reader's daily life (1-10)"},
        "summary": {"type": "string", "description": "3-4 sentence summary in the target language, slightly sarcastic or humorous where appropriate, ending with 1-3 emojis matching the tone"},
    },
    "required": ["expat_impact", "event_weight", "politics", "timeliness", "practical_utility", "summary"],
    "additionalProperties": False,
}


_EVALUATE_FEW_SHOT = """
Score each dimension 1-10 (politics is 0-10). Use these anchors:
- expat_impact: 1=only matters to native Spaniards (regional pension reform); 5=general interest; 10=directly affects expats (visas, flights, English services, foreign communities).
- event_weight: 1=routine bureaucracy; 5=notable local event; 10=rare or city-defining (major opening, disaster, milestone).
- politics: 0=pure party politics with no public-service angle; 5=policy with concrete citizen impact; 10=apolitical (innovation, science, culture, weather).
- timeliness: 1=evergreen; 5=this week; 10=happening now, action required today.
- practical_utility: 1=trivia; 5=nice to know; 10=changes what reader does today (closure, deadline, opportunity).

Examples:

<example>
<article>
Ryanair anuncia una nueva ruta directa entre el aeropuerto de Malaga-Costa del Sol y Berlin, con cuatro vuelos semanales a partir del 28 de octubre. Los billetes saldran a la venta esta semana desde 24,99 euros.
</article>
<output>{"expat_impact": 9, "event_weight": 7, "politics": 10, "timeliness": 8, "practical_utility": 9, "summary": "Ryanair is launching a direct Malaga-Berlin route with four weekly flights from 28 October. Tickets go on sale this week from a wallet-friendly 24.99 euros, so your excuses for skipping that Berlin weekend are officially gone. ✈️🇩🇪"}</output>
</example>

<example>
<article>
El PP de Malaga acusa al PSOE de "bloquear" la comision municipal de hacienda tras la ausencia de tres concejales socialistas en la sesion del martes. El portavoz socialista ha respondido calificando las declaraciones de "cortina de humo".
</article>
<output>{"expat_impact": 2, "event_weight": 2, "politics": 1, "timeliness": 4, "practical_utility": 1, "summary": "Malaga's PP accuses the PSOE of blocking the municipal finance committee after three socialist councillors skipped Tuesday's session. The socialists call it all a smoke screen — in other words, a perfectly ordinary day in local politics. 🎭"}</output>
</example>

<example>
<article>
La AEMET activa el aviso naranja por lluvias torrenciales en la provincia de Malaga para este jueves, con acumulados que podrian superar los 80 litros por metro cuadrado en cuatro horas. El 112 recomienda evitar desplazamientos no esenciales y los ayuntamientos de la costa cierran parques y playas.
</article>
<output>{"expat_impact": 8, "event_weight": 6, "politics": 10, "timeliness": 10, "practical_utility": 10, "summary": "AEMET has issued an orange alert for torrential rain in Malaga province this Thursday, with up to 80 litres per square metre possible in just four hours. Emergency services advise skipping non-essential trips, and coastal towns are closing parks and beaches — a good day to admire the weather from your sofa. 🌧️⚠️"}</output>
</example>
""".strip()


def get_evaluate_and_summarize_prompt(source_language='es', target_language='en'):
    source_lang = _lang_name(source_language)
    target_lang = _TARGET_LANG_STYLES.get(target_language.lower(), 'English')
    return (
        "You are a news evaluation agent. Your role is to score local news stories based on how likely they are to interest "
        f"a general audience, especially international readers and expats in Malaga, and to summarize them. The article is written in {source_lang}.\n\n"
        f"In the \"summary\" field, summarize the article with details in 3-4 sentences in {target_lang}, "
        "with a slightly sarcastic or humorous style where appropriate. End the summary with 1-3 emojis that match the tone of the news.\n\n"
        f"{_EVALUATE_FEW_SHOT}\n\n"
        "Now score and summarize the article that follows. Return ONLY the JSON object - no commentary, no markdown."
    )
