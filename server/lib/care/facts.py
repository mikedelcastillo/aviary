"""Verified bird-care knowledge for the aviary flock — embedded data.

These facts were produced by a multi-source research pass (2026-06-25) over
authoritative avian-care references — avian veterinarians and vet hospitals
(VCA, LafeberVet), the MSD/Merck Veterinary Manual, Cornell, RSPCA and species
societies. Every diet-toxicity, temperature, hazard and emergency ("safety
critical") claim was independently fact-checked against a second reputable
source before being kept, and over-strong claims were softened to the
consensus. The flock: lovebirds (Percy, Matcha, Jynx), a budgie (Bambi), and
cockatiels (Draft, Pizza).

This is curated reference content, not configuration — so it lives in code as
frozen data (like roster.DEFAULT_SPECIES_MEMBERS) rather than an env/YAML file:
no I/O, no parse-failure path, importable and unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CareFact:
    """One curated care fact. ``species`` is "general" or a species name."""

    species: str
    topic: str
    fact: str
    safety_critical: bool


# The recommended uninterrupted dark/sleep window per night (adult birds).
SLEEP_HOURS = (10, 12)
SLEEP_HOURS_TEXT = "10-12 hours of uninterrupted, dark, quiet sleep per night for the adult birds (lovebirds, budgie, and cockatiels), on a consistent schedule; baby cockatiels need 14-16 hours."
TEMPERATURE_RANGE_TEXT = "Comfortable range ~65-80°F (18-27°C) for all of these species, with steady temperature being the priority. Cockatiels do best around 70-75°F (21-23°C); budgies and lovebirds tolerate roughly 60-85°F. Cold danger threshold: below ~65°F (18°C) birds can become chilled/stressed and more prone to respiratory illness, and below ~50°F (10°C) hypothermia risk rises sharply. Avoid sudden swings greater than ~10-20°F over a short period and keep birds out of drafts."
# Below this ambient (°F) a small bird is at real risk (chilling -> illness).
COLD_STRESS_F = 65
COLD_DANGER_F = 50
# Perishable fresh produce should be pulled after roughly this long (food safety).
FRESH_FOOD_MINUTES = 120

CARE_SUMMARY = "This flock — lovebirds Percy, Matcha and Jynx, budgie Bambi, and cockatiels Draft and Pizza — thrives on consistency and a pellet-based diet: make 75-80% formulated pellets the foundation (60-80% for Bambi), with washed, small-cut greens and veggies as the bulk of the rest, fruit only sparingly, and minimal seed; never offer grit. Keep every toxic food (avocado, chocolate, caffeine, alcohol, onion/garlic, fruit pits and apple seeds, xylitol, salty/fatty/sugary scraps) completely out of reach, and protect their sensitive lungs from Teflon fumes, aerosols, candles and smoke. Anchor their day to observed lights-on and PH sunrise/sunset: fresh food and water each morning, fresh produce pulled after ~2 hours, midday enrichment and out-of-cage time, a calm wind-down before a consistent bedtime, and a reliable 10-12 hours of dark — with a dim red/amber nightlight for the cockatiels, who are highly prone to injurious night frights. Hold the room around 65-80°F with no drafts or sudden swings. Weigh each bird weekly before breakfast and log the trend, because birds hide illness and a ~10% drop, refused food, tail-bobbing or open-mouth breathing means an avian vet now, not later. The biggest caveat for mixed housing is the lovebirds: they are intensely territorial and aggressive (hens most of all, especially around nests) and will attack birds of any size, so do not house them with the budgie or cockatiels in a standard cage — use a large walk-in or divided aviary with multiple separated food/water stations and retreat zones, introduce cautiously, and never leave them unsupervised together."


# -- the knowledge base ------------------------------------------------------

CARE_FACTS: tuple[CareFact, ...] = (
    CareFact("general", "diet", "Feed a base diet of 75-80% formulated pellets, with no more than ~20-25% fresh vegetables/greens and limited fruit, and keep seeds to a minimal amount. A seed-only diet causes malnutrition because seeds are low in vitamins, minerals and protein and high in fat.", False),
    CareFact("general", "diet", "Favor dark leafy greens and vegetables (broccoli, carrots, bell peppers, squash, peas, cooked sweet potato) over fruit; offer fruit only sparingly because of its high water and natural sugar content.", False),
    CareFact("general", "diet", "Wash all fresh foods thoroughly, cut them into small bird-sized pieces, and remove uneaten produce promptly — roughly within 2 hours (sooner in warm rooms) — to prevent bacterial spoilage. The ~2-hour window is a conservative food-safety guideline rather than a hard rule.", True),
    CareFact("general", "diet", "Do NOT offer insoluble grit or gravel. These parrots (cockatiels, lovebirds, budgies) all hull their seeds before swallowing and have no digestive need for grit; over-consuming it can cause life-threatening crop or gastrointestinal impaction. Soluble mineral sources like cuttlebone or oyster shell are a separate, safe category.", True),
    CareFact("general", "diet", "Birds on a balanced 75-80% pelleted diet generally do not need vitamin/mineral supplements. Supplements coated on seed are ineffective because birds de-hull seeds before eating. Provide a cuttlebone or mineral block for calcium; egg-laying hens need extra calcium to help prevent egg-binding.", False),
    CareFact("general", "diet", "Convert a seed-dependent bird to pellets gradually, commonly over ~7-14 days using staged ratios (e.g. 75/25 to 50/50 to 25/75 seed-to-pellet). Never simply remove all seed at once for an unmonitored bird — birds can starve rather than eat unfamiliar food. Weigh daily, watch droppings (small/dark droppings signal it isn't eating), and consult an avian vet if it stops eating or loses weight.", True),
    CareFact("general", "sleep_light", "Provide 10-12 hours of uninterrupted, dark, quiet sleep every night on a consistent schedule; baby cockatiels need 14-16 hours. A regular bedtime is what makes the dark window work. Cover the cage or use room-darkening blinds if household light extends past bedtime.", False),
    CareFact("general", "sleep_light", "Aim for ~10-12 hours of daytime light, ideally including some natural/full-spectrum exposure. Window glass blocks UV, so unobstructed outdoor skylight or a full-spectrum bird lamp is beneficial. Limit total light to no more than 12 hours per day.", False),
    CareFact("general", "sleep_light", "Excess light and long days stimulate breeding hormones, which can trigger problematic chronic egg-laying in females and increased aggression. Keeping nights long and dark (~12h) and removing nesting cues helps prevent hormonal behavior.", False),
    CareFact("general", "enrichment", "Provide varied, rotating toys (foraging, chew, shreddable) made of bird-safe wood, paper, cardboard, rope, or firm plastic to prevent boredom, which can lead to screaming and feather plucking.", False),
    CareFact("general", "enrichment", "Give supervised daily out-of-cage time for exercise and mental stimulation (cockatiels ~1-3 hours, budgies at least ~1 hour). Before flight time, cover windows, secure other pets, and remove hazards (open water, fans, toxic items). Never leave the bird unattended out of the cage.", False),
    CareFact("general", "social", "These are highly social flock animals that need daily interaction with their people or a compatible companion bird. A solitary, under-stimulated bird can become depressed, scream, or pluck. Keep budgies in pairs/groups or provide hours of daily human contact.", False),
    CareFact("general", "social", "For any shared/aviary space, provide multiple widely separated food and water stations plus visual barriers and retreat zones, so a dominant bird cannot guard the only food source and bullied birds can escape. This directly reduces resource-guarding conflict.", False),
    CareFact("general", "social", "Use a large walk-in or divided aviary, not a standard cage, for any mixed-species housing. A standard aviary holding ~10 budgies or 2 cockatiels is too small even for 3 budgies plus 2 cockatiels; mixed groups need room to fly away from conflict, and a divider lets you separate species in the same space.", False),
    CareFact("general", "hygiene", "Daily cage care: replace the disposable paper liner, remove uneaten food, droppings and debris, and change to fresh water every day. Check rope/fabric toys daily for loose strings that could entangle the bird.", False),
    CareFact("general", "hygiene", "Weekly: wash perches, toys, and dishes separately in hot water with bird-safe soap, and replace bedding/substrate at least once a week (more often with multiple birds).", False),
    CareFact("general", "hygiene", "Provide a regular bathing opportunity (misting/spray or a shallow bath); keep cages in a well-lit, draft-free, well-ventilated spot, off the floor, and away from other pets.", False),
    CareFact("general", "health_signs", "Birds instinctively hide illness; by the time symptoms show, the bird is often seriously ill, so act fast. Warning signs include persistently fluffed feathers, lethargy/excess sleeping with eyes closed, sitting low on the perch or cage floor, weakness/loss of balance, reduced vocalization, and changes in appetite, thirst, or droppings.", True),
    CareFact("general", "health_signs", "Tail-bobbing with each breath, wheezing, or open-mouth breathing signals respiratory distress — an emergency requiring urgent avian-veterinary care.", True),
    CareFact("general", "health_signs", "Weigh each bird weekly on a gram-accurate (1-g) scale, ideally in the morning before feeding, same perch/container, and log it to learn its normal range. A trend matters more than one number, and weight loss is often the earliest detectable clue of hidden illness.", True),
    CareFact("general", "health_signs", "Seek prompt avian-vet care for a ~10% drop in body weight (faster loss is more urgent), refusing food for more than ~24 hours, or 24+ hours of unusual lethargy/silence/fluffing — especially when combined with other signs.", True),
    CareFact("general", "emergency", "Keep a sick or chilled bird warm (around 85-90°F / 29-32°C) and quiet while arranging immediate avian-vet care. Ensure the bird can move to a cooler spot to avoid overheating, warm gradually, and use heat cautiously in suspected head-trauma cases.", True),
    CareFact("general", "temperature", "Keep the cage out of drafts and away from windows, radiators, and air conditioners. Cold drafts and rapid temperature swings can chill a bird and contribute to illness; covering the cage at night helps reduce drafts.", True),
    CareFact("general", "temperature", "Protect birds from airborne fumes: overheated nonstick (PTFE/Teflon) cookware fumes can be fatal within minutes. Aerosol sprays, scented candles, air fresheners, incense, and smoke are also dangerous to birds' highly sensitive respiratory systems.", True),
    CareFact("cockatiel", "hygiene", "Cockatiels are a dusty powder-down species. Mist/spray-bathe them regularly (without soaking) to maintain healthy skin and feathers; regular bathing plus good ventilation or an air purifier near the cage manages airborne dust for both bird and human respiratory health.", False),
    CareFact("cockatiel", "health_signs", "A healthy cockatiel is alert, active, vocal, eats well, has smooth feathers, clear eyes/nostrils, and well-formed green-and-white droppings. Preening is normal healthy behavior.", False),
    CareFact("cockatiel", "health_signs", "Cockatiels are the species most prone to night frights — sudden panicked thrashing in the dark that can cause broken blood feathers, damaged wings/eyes, or fractures. A dim red/amber nightlight, secure perches, and a quiet sleeping location help them orient and reduce panicked thrashing.", True),
    CareFact("cockatiel", "emergency", "Egg-binding (dystocia) is a cockatiel emergency: a hen straining, fluffed, lethargic, sitting on the cage floor, with abdominal swelling or labored breathing needs immediate avian-vet care. Affected birds can deteriorate and die within hours — do not wait.", True),
    CareFact("cockatiel", "diet", "Cockatiels on a 75-80% pelleted diet do not need vitamin/mineral supplements, but egg-laying hens need extra calcium (cuttlebone) to help prevent egg-binding and deficiency.", False),
    CareFact("lovebird", "diet", "Lovebirds do well with pellets as 75-80% of the diet and 20-40% fresh fruits/vegetables/greens. Offer fresh foods in tiny portions (a thumbnail-sized piece is a full serving for a bird) and remove perishable fresh food after ~2 hours.", False),
    CareFact("lovebird", "social", "Lovebirds reach sexual maturity at 8-12 months, bringing hormonal behavior: increased vocalization, territorial aggression, nesting (hens tuck nest material under wing/tail feathers), and regurgitation. Reduce light to ~12h, remove nest-like spaces and shredded-paper hideaways, and avoid petting the back/under wings to calm it.", False),
    CareFact("lovebird", "social", "Single lovebirds bond closely to their owner but need substantial daily interaction as a substitute for a mate. Bonded pairs often bond more strongly to each other than to people and become less hand-tame.", False),
    CareFact("lovebird", "social", "Lovebirds are very territorial and aggressive — especially when frightened, threatened, or bonded to a person — and will threaten or attack other birds regardless of size, even chasing larger birds, dogs, or cats. Do not house different lovebird species or unrelated bird species together; introduce new birds cautiously and never leave lovebirds unsupervised with other birds or pets.", True),
    CareFact("lovebird", "enrichment", "Lovebirds are active, destructive chewers and avid paper shredders. Provide abundant bird-safe soft wood, dye-free/plain paper, seed bells, swings, ladders, and wooden gnaws, and rotate toys to prevent boredom and feather plucking.", False),
    CareFact("lovebird", "temperature", "A fluffed-up bird held that way for long periods is likely cold; a bird holding its wings out/drooping and panting is overheated. Warm a chilled bird and offer a cool environment to an overheated one. Prolonged fluffing with lethargy can also signal illness or hypothermia — contact an avian vet.", True),
    CareFact("budgie", "diet", "Make pellets 60-80% of a budgie's diet; limit fresh vegetables, greens, and fruit to ~20-25%, cut small and washed. Budgies are prone to iodine-deficiency goiter on seed-heavy diets — converting to pellets is the most effective prevention; an avian vet can advise on iodine if needed.", False),
    CareFact("budgie", "social", "Budgies are flock animals for whom social contact with their own kind is vital — a solitary budgie's life is described as an evolutionary aberration. Keep them in pairs/groups for mutual preening and chattering, or provide hours of daily human interaction if kept alone.", False),
    CareFact("budgie", "toxic_foods", "Budgerigars are among the birds most susceptible to avocado: as little as ~8.7 g of mashed fruit has been fatal within 48 hours. All parts of the avocado plant contain persin, causing heart-muscle necrosis, breathing difficulty, weakness, and sudden death. Contact an avian or emergency vet immediately if any avocado is eaten.", True),
    CareFact("budgie", "temperature", "For budgies, below ~65°F (18°C) they can become chilled and stressed and more prone to respiratory illness; below ~50°F (10°C) hypothermia risk rises sharply. Early cold signs: persistent feather fluffing (outside normal rest), shivering, tucking a foot up, lethargy. Warm gradually (to ~80-90°F) and contact an avian vet if hypothermic; avoid sudden direct heat.", True),
)


# -- toxic / dangerous foods -------------------------------------------------


@dataclass(frozen=True)
class ToxicFood:
    """A food that is toxic or dangerous to these birds.

    ``aliases`` are the lowercase phrases to match in a user's message; they are
    matched on word boundaries so e.g. "cola" never fires inside "chocolate".
    """

    name: str
    aliases: tuple[str, ...]
    reason: str


# Hand-curated from the verified research (aliases tuned to avoid false matches).
# Safety-critical: this list backs the bot answering "can the birds eat X?".
TOXIC_FOODS: tuple[ToxicFood, ...] = (
    ToxicFood(
        "avocado",
        ("avocado", "avocados", "guacamole", "guac"),
        "all parts contain persin, which causes fatal heart-muscle damage; budgies are especially susceptible (a tiny amount can kill within a day or two)",
    ),
    ToxicFood(
        "chocolate",
        ("chocolate", "cocoa", "cacao"),
        "theobromine and caffeine cause tremors, seizures, a racing heart, and can be fatal",
    ),
    ToxicFood(
        "caffeine",
        ("caffeine", "coffee", "espresso", "energy drink"),
        "causes arrhythmias, seizures, and cardiac arrest in small birds",
    ),
    ToxicFood(
        "alcohol",
        ("alcohol", "beer", "wine", "liquor", "whiskey", "vodka"),
        "causes central-nervous-system depression and organ failure — never offer it",
    ),
    ToxicFood(
        "onion",
        ("onion", "onions", "scallion", "shallot", "leek"),
        "sulfur compounds cause hemolytic anemia (raw, cooked, or powdered)",
    ),
    ToxicFood(
        "garlic",
        ("garlic",),
        "the same organosulfur compounds as onion damage red blood cells; powdered/concentrated forms are worst",
    ),
    ToxicFood(
        "apple seeds",
        ("apple seed", "apple seeds", "apple pip", "apple pips"),
        "contain cyanogenic amygdalin (cyanide) — the flesh is fine once the seeds are removed",
    ),
    ToxicFood(
        "fruit pits",
        # Only the PIT/seed/core is toxic — the fruit flesh (cherry, peach, plum…)
        # is safe, so bare fruit names are deliberately NOT aliases (they'd wrongly
        # trip the safety gate on "did they eat the cherries").
        ("fruit pit", "fruit pits", "stone fruit", "cherry pit", "peach pit", "apricot pit", "plum pit", "apple core"),
        "cherry/plum/apricot/peach pits release cyanide; the depitted fruit flesh itself is safe",
    ),
    ToxicFood(
        "xylitol",
        ("xylitol", "sugar-free gum", "sugar free gum", "sugarfree"),
        "this sweetener (in sugar-free gum/candy/diet foods) causes hypoglycemia and liver damage",
    ),
    ToxicFood(
        "salty foods",
        ("salt", "salty", "salted", "potato chips", "chips", "pretzel", "pretzels"),
        "excess salt causes electrolyte imbalance, dehydration, and kidney damage in small birds",
    ),
    ToxicFood(
        "dairy",
        ("milk", "cheese", "dairy", "yogurt", "butter", "ice cream"),
        "birds can't digest lactose; avoid it, or give only tiny amounts of low-lactose items",
    ),
    ToxicFood(
        "fatty/fried foods",
        ("fried food", "fried foods", "fast food", "junk food", "chips and"),
        "high-fat, high-sugar processed/fried foods lead to obesity, fatty liver, and atherosclerosis",
    ),
)

