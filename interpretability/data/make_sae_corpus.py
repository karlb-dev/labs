"""Generate the frozen feature-interpretation corpus for Lab 8.

Run once at authoring time; the CSV is vendored and never regenerated at lab
runtime (course rule: no live data downloads). Deterministic — no RNG.

The corpus is the substrate the SAE is run over to find, label, and *validate*
features. Its design is the whole point of the lab, so it is built to make
label validation possible rather than just to be "diverse text":

  * Every line carries a ``domain`` tag (chemistry, sports, code, ...). Domain
    membership is the held-out ground truth a proposed label is tested against:
    if you label a feature "fires on chemistry," the validation battery checks
    whether it fires on held-out chemistry lines and stays quiet on the rest.

  * Domains come in *confusable pairs* by surface token, on purpose, so the
    corpus can distinguish a semantic feature from a lexical one:
      - chemistry vs. cooking      (both: "acid", "base", "salt", "reduce")
      - finance vs. sports         (both: "score", "lead", "beat", "rally")
      - law vs. medicine           (both: "trial", "discharge", "administer")
      - weather vs. emotion        (both: "storm", "cloud", "bright", "cold")
    A feature that fires on the WORD "acid" in both chemistry and cooking is a
    token feature; one that fires on chemistry "acid" but not cooking "acid" is
    a concept feature. The lab's adversarial near-miss prompts are drawn from
    these pairs.

  * A handful of lines are deliberately ambiguous (a sentence that is half
    finance, half sports) to surface polysemantic features honestly.

Output:
  sae_feature_corpus.csv  -- columns: text_id, domain, text
"""

from __future__ import annotations

import csv
import pathlib

HERE = pathlib.Path(__file__).parent

# Each domain: a list of short, self-contained lines. Kept short (one or two
# sentences) so a single SAE forward pass over the line is cheap and the
# top-activating *context* is legible. ~16 lines/domain x 10 domains ~= 160.
CORPUS: dict[str, list[str]] = {
    "chemistry": [
        "The strong acid donated a proton to the base, forming a salt and water.",
        "Titrating the solution to its equivalence point turned the indicator pink.",
        "Oxidation strips electrons from the metal while the cathode is reduced.",
        "A buffer resists changes in pH when small amounts of acid are added.",
        "The reaction was exothermic, releasing heat as the bonds reorganized.",
        "Sodium chloride dissolves into its constituent ions in aqueous solution.",
        "Catalysts lower the activation energy without being consumed themselves.",
        "The molar mass determines how many grams make up one mole of the compound.",
        "Electrons fill orbitals from the lowest available energy level upward.",
        "Distillation separates the mixture by exploiting differences in boiling point.",
        "The precipitate formed the moment the two clear solutions were combined.",
        "Covalent bonds share electron pairs between adjacent nonmetal atoms.",
        "Increasing the concentration of reactants shifts the equilibrium forward.",
        "The pH meter read 2.1, confirming the sample was strongly acidic.",
        "Enzymes are biological catalysts that accelerate metabolic reactions.",
        "The gas expanded to fill the flask as the temperature climbed.",
    ],
    "cooking": [
        "Whisk the acid from a squeeze of lemon into the warm butter sauce.",
        "Reduce the stock over high heat until it coats the back of a spoon.",
        "Season the base of the soup with salt before adding the cream.",
        "Let the dough rest so the gluten relaxes and the crust stays tender.",
        "Sear the steak hard on one side to build a deep brown crust.",
        "Fold the beaten egg whites gently so the batter keeps its air.",
        "A pinch of sugar balances the acid in an over-tart tomato sauce.",
        "Caramelize the onions slowly until they turn sweet and jammy.",
        "Rest the roast under foil so the juices redistribute before carving.",
        "Toast the spices in a dry pan to wake up their fragrance.",
        "Deglaze the pan with wine, scraping up the browned bits for flavor.",
        "Blanch the greens briefly, then shock them in ice water to set the color.",
        "Knead the dough until it springs back when you press it with a finger.",
        "Temper the chocolate so it sets with a glossy snap.",
        "Salt the pasta water generously; it should taste like the sea.",
        "Simmer the curry until the sauce thickens and the oil rises to the top.",
    ],
    "sports": [
        "She dribbled past two defenders and scored in the final minute of the match.",
        "The rookie's late rally pulled the team back to within a single point.",
        "He beat the throw to second base with a headfirst slide.",
        "The striker's header found the top corner to seal the win.",
        "Down by ten, the home side mounted a furious comeback in the fourth quarter.",
        "The goalkeeper dove full stretch to tip the shot around the post.",
        "Their captain led the break and finished with a thunderous dunk.",
        "The marathon leader pulled away on the final hill and never looked back.",
        "A clean tackle stripped the ball just outside the penalty area.",
        "The pitcher struck out the side to escape the bases-loaded jam.",
        "The peloton chased hard but could not reel in the lone breakaway rider.",
        "She served three aces in a row to take the deciding set.",
        "The forwards pressed high and forced a turnover deep in the zone.",
        "His buzzer-beater from half court sent the arena into a frenzy.",
        "The relay team shaved a full second off the national record.",
        "A perfectly timed block kept the spiker from scoring the final point.",
    ],
    "finance": [
        "Quarterly revenue rose twelve percent, and the stock rallied on the beat.",
        "The bond's yield climbed as investors priced in another rate hike.",
        "Their lead over rivals narrowed as margins compressed across the sector.",
        "The fund rebalanced into equities after scoring gains in fixed income.",
        "Cash flow turned positive once the company cut its operating costs.",
        "The merger was financed with a mix of debt and new share issuance.",
        "Analysts upgraded the shares, citing a strong order backlog.",
        "Inflation data sent the index sharply lower in afternoon trading.",
        "The startup raised a Series B at a valuation north of one billion.",
        "Dividends were held flat while the board prioritized debt reduction.",
        "A short squeeze drove the heavily shorted stock up forty percent.",
        "The central bank held rates steady but signaled cuts later in the year.",
        "Earnings beat estimates, yet guidance for next quarter disappointed.",
        "The portfolio's beta rose as it tilted toward cyclical names.",
        "Currency losses ate into the exporter's reported profit this period.",
        "The IPO priced at the top of its range amid heavy demand.",
    ],
    "law": [
        "The court granted the motion to dismiss for lack of jurisdiction.",
        "At trial, the defense argued the evidence had been unlawfully obtained.",
        "The statute requires the plaintiff to file within two years of the injury.",
        "The judge instructed the jury to disregard the stricken testimony.",
        "Under the contract, either party may terminate with thirty days' notice.",
        "The appellate panel reversed and remanded the case for a new hearing.",
        "Precedent binds the lower court to follow the higher court's ruling.",
        "The witness was administered the oath before taking the stand.",
        "The settlement released both parties from any further liability.",
        "The prosecution must prove every element beyond a reasonable doubt.",
        "The clause was held unenforceable as contrary to public policy.",
        "Discovery obligations require each side to disclose relevant documents.",
        "The defendant entered a plea of not guilty at the arraignment.",
        "An injunction barred the company from using the disputed trademark.",
        "The deposition was transcribed and entered into the record.",
        "The tribunal ruled the arbitration clause governed the dispute.",
    ],
    "medicine": [
        "The patient was administered antibiotics and discharged the next morning.",
        "A clinical trial found the drug reduced relapse rates by a third.",
        "Elevated troponin pointed to damage to the heart muscle.",
        "The surgeon resected the tumor and sent margins to pathology.",
        "Symptoms included fever, fatigue, and a persistent dry cough.",
        "The dose was titrated upward until the seizures were controlled.",
        "An MRI revealed a small lesion in the left temporal lobe.",
        "The vaccine primes the immune system to recognize the pathogen.",
        "Blood pressure remained elevated despite the new medication.",
        "The biopsy confirmed the growth was benign rather than malignant.",
        "Physical therapy restored most of the range of motion in the joint.",
        "The chart noted an allergy to penicillin in bold red ink.",
        "Insulin lowers blood glucose by moving it into the cells.",
        "The wound was sutured and dressed to prevent infection.",
        "The cardiologist recommended a stress test before clearing him.",
        "Chemotherapy shrank the mass enough to make surgery possible.",
    ],
    "weather": [
        "The storm pushed inland overnight, dropping three inches of rain.",
        "A bright, cloudless sky followed the cold front through the valley.",
        "Gusts topped sixty miles an hour as the squall line passed.",
        "Dense fog cut visibility to a few hundred feet on the coast road.",
        "The heat dome parked over the region for a sixth straight day.",
        "Snow tapered to flurries as the temperature crept above freezing.",
        "Lightning forked across the horizon ahead of the approaching cell.",
        "A gentle drizzle settled in, leaving the streets slick and gray.",
        "The barometer fell sharply, a sign the weather was about to turn.",
        "Clear skies tonight will let temperatures dip near the frost point.",
        "The hurricane weakened to a tropical storm after making landfall.",
        "Hail the size of marbles dented cars across the suburb.",
        "A warm breeze carried the smell of rain before the first drops fell.",
        "Black clouds massed on the ridge as the afternoon storm built.",
        "The drought broke at last with a slow, soaking two-day rain.",
        "Ice glazed the branches and brought down power lines overnight.",
    ],
    "emotion": [
        "A cold dread settled over her as the footsteps grew louder.",
        "His face brightened the moment he saw the familiar handwriting.",
        "Grief came in waves, quiet one hour and overwhelming the next.",
        "She felt a warm rush of pride watching her daughter cross the stage.",
        "The room was thick with tension, every word measured and wary.",
        "Relief washed over him when the test results finally came back clear.",
        "A storm of anger rose in her chest, then slowly, deliberately cooled.",
        "He carried a dull, lingering sadness he could not quite name.",
        "Joy bubbled up so suddenly she laughed before she could speak.",
        "Loneliness pressed in on the long, silent drive home.",
        "Hope flickered, small and stubborn, against everything stacked against them.",
        "The shame burned hot, and he wished the floor would swallow him.",
        "Her heart lifted at the bright, unexpected kindness of a stranger.",
        "A heavy gloom hung over the house in the weeks after the loss.",
        "Excitement crackled through the crowd as the lights went down.",
        "Calm returned slowly, like warmth seeping back into cold hands.",
    ],
    "code": [
        "def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
        "for i in range(len(items)):\n    total += items[i].price * items[i].qty",
        "class Stack:\n    def push(self, x):\n        self._data.append(x)",
        "result = [x * 2 for x in numbers if x % 2 == 0]",
        "try:\n    conn = connect(host)\nexcept TimeoutError:\n    retry(conn)",
        "const sum = arr.reduce((acc, x) => acc + x, 0);",
        "SELECT name, COUNT(*) FROM orders GROUP BY name HAVING COUNT(*) > 3;",
        "while not queue.empty():\n    node = queue.get()\n    visit(node)",
        "git rebase -i HEAD~3 && git push --force-with-lease origin feature",
        "async def fetch(url):\n    async with session.get(url) as r:\n        return await r.json()",
        "if (ptr == NULL) { return -1; }\n*ptr = value;",
        "import numpy as np\nA = np.zeros((3, 3))\nA[np.diag_indices(3)] = 1",
        "def binary_search(a, t):\n    lo, hi = 0, len(a) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2",
        "@app.route('/users/<int:uid>')\ndef get_user(uid):\n    return jsonify(db.find(uid))",
        "func main() {\n    ch := make(chan int)\n    go worker(ch)\n}",
        "type Point = { x: number; y: number };\nconst origin: Point = { x: 0, y: 0 };",
    ],
    "history": [
        "The Treaty of Westphalia in 1648 ended the Thirty Years' War in Europe.",
        "Rome fell to the Visigoths in 410, a shock felt across the ancient world.",
        "The printing press spread literacy faster than any prior invention.",
        "Napoleon's retreat from Moscow in 1812 shattered his Grande Armée.",
        "The Magna Carta limited the king's power and seeded later constitutions.",
        "The transcontinental railroad joined the coasts of America in 1869.",
        "The Renaissance revived classical art and learning across Italy.",
        "The fall of the Berlin Wall in 1989 hastened the end of the Cold War.",
        "Gutenberg's Bible was among the first books printed with movable type.",
        "The Silk Road carried goods and ideas between China and the Mediterranean.",
        "The French Revolution toppled the monarchy and proclaimed a republic.",
        "The discovery of the New World reshaped trade and empire for centuries.",
        "The Industrial Revolution pulled workers from farms into crowded cities.",
        "Alexander's conquests spread Greek culture as far east as India.",
        "The signing of the Declaration in 1776 announced a new nation.",
        "The plague of the fourteenth century killed a third of Europe.",
    ],
}

# A few deliberately mixed lines: each blends two domains so a genuinely
# polysemantic or domain-confused feature has something honest to fire on.
# Tagged with the compound domain so they are excluded from single-domain
# held-out validation (the lab filters these out of the ground-truth sets).
MIXED: list[tuple[str, str]] = [
    ("finance+sports", "The team's stock rallied after they beat their rivals to clinch the title."),
    ("chemistry+cooking", "Balancing the acid and the base is chemistry whether in a flask or a sauce."),
    ("law+medicine", "At trial the surgeon testified about the dose she had administered."),
    ("weather+emotion", "A cold dread settled in as the storm clouds massed on the horizon."),
    ("history+finance", "The new central bank, founded in 1694, financed the kingdom's wars."),
    ("code+history", "The first compiler, written in the 1950s, changed how software was built."),
]


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    n = 0
    for domain in sorted(CORPUS):
        for text in CORPUS[domain]:
            n += 1
            rows.append({"text_id": f"T{n:03d}", "domain": domain, "text": text})
    for domain, text in MIXED:
        n += 1
        rows.append({"text_id": f"T{n:03d}", "domain": domain, "text": text})
    return rows


def main() -> None:
    rows = build_rows()
    path = HERE / "sae_feature_corpus.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text_id", "domain", "text"])
        writer.writeheader()
        writer.writerows(rows)
    domains = sorted({r["domain"] for r in rows if "+" not in r["domain"]})
    print(f"wrote {len(rows)} lines across {len(domains)} single domains "
          f"+ {sum('+' in r['domain'] for r in rows)} mixed -> {path}")


if __name__ == "__main__":
    main()
