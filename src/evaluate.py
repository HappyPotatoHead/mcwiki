"""Retrieval evaluation: hit@k and MRR, cosine-only vs reranked."""

from collections import Counter

from .config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEVICE,
    EMBED_MODEL,
    QUERY_INSTRUCTION,
    RERANK_MODEL,
    RETRIEVE_TOP_N,
)
from .embedder import Embedder
from .retrieve import Match, Reranker
from .store import ChromaStore

DEBUG_QUERIES = {
    "What do I need to make a Nether portal?",
    "What enchantment makes tools durability last longer?",
    "How do I make bread?",
}


TEST_CASES = [
    # Crafting and basic survival (1-10)
    {
        "query": "What do I need to make a Nether portal?",
        "relevant_titles": ["Nether Portal", "Obsidian"],
    },
    {
        "query": "How do I make a furnace?",
        "relevant_titles": ["Furnace", "Cobblestone"],
    },
    {
        "query": "How do I craft a crafting table?",
        "relevant_titles": ["Crafting", "Planks"],
    },
    {
        "query": "How do I make a chest?",
        "relevant_titles": ["Chest", "Planks"],
    },
    {
        "query": "How do I make a bed?",
        "relevant_titles": ["Bed", "Wool", "Planks"],
    },
    {
        "query": "How do I make bread?",
        "relevant_titles": ["Bread", "Wheat"],
    },
    {
        "query": "What tools do I need to mine diamond ore?",
        "relevant_titles": ["Diamond", "Diamond Ore", "Pickaxe"],
    },
    {
        "query": "How do I make torches?",
        "relevant_titles": ["Torch", "Coal", "Charcoal"],
    },
    {
        "query": "How do I make a shield?",
        "relevant_titles": ["Shield", "Iron Ingot", "Planks"],
    },
    {
        "query": "How do I craft an anvil?",
        "relevant_titles": ["Anvil", "Iron Ingot", "Block of Iron"],
    },
    # Enchantment and equipment (11-20)
    {
        "query": "What enchantment makes tools durability last longer?",
        "relevant_titles": ["Unbreaking", "Durability", "Mending"],
    },
    {
        "query": "What does Mending do?",
        "relevant_titles": ["Mending", "Experience"],
    },
    {
        "query": "What does Silk Touch do?",
        "relevant_titles": ["Silk Touch", "Enchantment"],
    },
    {
        "query": "What does Fortune do?",
        "relevant_titles": ["Fortune", "Enchantment"],
    },
    {
        "query": "How do I enchant a weapon?",
        "relevant_titles": ["Enchantment", "Enchanting Table", "Anvil"],
    },
    {
        "query": "How can I repair damaged tools?",
        "relevant_titles": ["Anvil", "Grindstone", "Mending"],
    },
    {
        "query": "How do I combine enchanted books with equipment?",
        "relevant_titles": ["Anvil", "Enchantment"],
    },
    {
        "query": "How do I make stronger armour?",
        "relevant_titles": ["Armor", "Smithing", "Enchantment"],
    },
    {
        "query": "How do I upgrade diamond equipment to netherite?",
        "relevant_titles": ["Block of Netherite", "Smithing", "Smithing Template"],
    },
    {
        "query": "What is the difference between a sword and an axe in combat?",
        "relevant_titles": ["Sword", "Axe", "Damage"],
    },
    # Villagers and villages (21-30)
    {
        "query": "How do villagers reproduce?",
        "relevant_titles": ["Villager", "Breeding"],
    },
    {
        "query": "How do I breed villagers?",
        "relevant_titles": ["Villager", "Breeding", "Bed"],
    },
    {
        "query": "What is a librarian?",
        "relevant_titles": ["Librarian", "Villager", "Trading"],
    },
    {
        "query": "How do villagers get their professions?",
        "relevant_titles": ["Villager", "Workstation", "Trading"],
    },
    {
        "query": "How do I trade with villagers?",
        "relevant_titles": ["Trading", "Villager", "Emerald"],
    },
    {
        "query": "How do I cure a zombie villager?",
        "relevant_titles": ["Zombie Villager", "Potion of Weakness", "Golden Apple"],
    },
    {
        "query": "How do I protect a village from raids?",
        "relevant_titles": ["Raid", "Village", "Iron Golem"],
    },
    {
        "query": "How do iron golems spawn?",
        "relevant_titles": ["Iron Golem", "Village", "Villager"],
    },
    {
        "query": "How can I find a village?",
        "relevant_titles": ["Village", "Villager"],
    },
    {
        "query": "How do I get better prices from villagers?",
        "relevant_titles": ["Trading", "Hero of the Village", "Zombie Villager"],
    },
    # Farming and food (31-40)
    {
        "query": "How does a wheat farm work?",
        "relevant_titles": ["Wheat", "Farmland", "Water"],
    },
    {
        "query": "How do I grow carrots and potatoes?",
        "relevant_titles": ["Carrot", "Potato", "Farmland"],
    },
    {
        "query": "How do I grow sugar cane?",
        "relevant_titles": ["Sugar Cane", "Water"],
    },
    {
        "query": "How do I grow pumpkins and melons?",
        "relevant_titles": ["Pumpkin", "Melon", "Farmland"],
    },
    {
        "query": "How can I make crops grow faster?",
        "relevant_titles": ["Bone Meal", "Farmland", "Wheat"],
    },
    {
        "query": "How do I turn rotten food or plants into compost?",
        "relevant_titles": ["Composter", "Bone Meal"],
    },
    {
        "query": "How do I breed cows for food?",
        "relevant_titles": ["Cow", "Breeding", "Wheat"],
    },
    {
        "query": "How do I breed chickens?",
        "relevant_titles": ["Chicken", "Breeding", "Egg"],
    },
    {
        "query": "How do I get honey from bees?",
        "relevant_titles": ["Bee", "Beehive", "Honey Bottle"],
    },
    {
        "query": "How do I avoid starving in survival mode?",
        "relevant_titles": ["Hunger (effect)", "Food", "Saturation (effect)"],
    },
    # Nether and brewing (41-50)
    {
        "query": "What is the Nether?",
        "relevant_titles": ["The Nether", "Nether Portal"],
    },
    {
        "query": "How do I enter the Nether?",
        "relevant_titles": ["The Nether", "Nether Portal", "Obsidian"],
    },
    {
        "query": "How do I brew a potion of weakness?",
        "relevant_titles": ["Potion of Weakness", "Brewing", "Fermented Spider Eye"],
    },
    {
        "query": "How do I make a brewing stand?",
        "relevant_titles": ["Brewing Stand", "Blaze Rod"],
    },
    {
        "query": "What do I need to brew potions?",
        "relevant_titles": ["Brewing", "Brewing Stand", "Nether Wart"],
    },
    {
        "query": "How do I make a potion last longer?",
        "relevant_titles": ["Brewing", "Redstone Dust"],
    },
    {
        "query": "How do I turn a potion into a splash potion?",
        "relevant_titles": ["Splash Potion", "Brewing", "Gunpowder"],
    },
    {
        "query": "Where can I find blaze rods?",
        "relevant_titles": ["Blaze", "Blaze Rod", "Nether Fortress"],
    },
    {
        "query": "How do piglins trade with players?",
        "relevant_titles": ["Piglin", "Bartering", "Gold Ingot"],
    },
    {
        "query": "Where can I find ancient debris?",
        "relevant_titles": ["Ancient Debris", "Block of Netherite", "The Nether"],
    },
    # The End and progression (51-60)
    {
        "query": "What is the End?",
        "relevant_titles": ["The End", "End Portal"],
    },
    {
        "query": "How do I find a stronghold?",
        "relevant_titles": ["Stronghold", "Eye of Ender", "End Portal"],
    },
    {
        "query": "How do I activate an End portal?",
        "relevant_titles": ["End Portal", "Eye of Ender", "Stronghold"],
    },
    {
        "query": "How do I defeat the Ender Dragon?",
        "relevant_titles": ["Ender Dragon", "End Crystal", "The End"],
    },
    {
        "query": "How do I respawn the Ender Dragon?",
        "relevant_titles": ["Ender Dragon", "End Crystal", "The End"],
    },
    {
        "query": "What does the dragon egg do?",
        "relevant_titles": ["Dragon Egg", "Ender Dragon"],
    },
    {
        "query": "How do I find an End city?",
        "relevant_titles": ["End City", "End City/Structure/Ship", "The End"],
    },
    {
        "query": "How do I get an Elytra?",
        "relevant_titles": ["Elytra", "End City/Structure/Ship", "End City"],
    },
    {
        "query": "How do I fly using an Elytra?",
        "relevant_titles": ["Elytra", "Firework Rocket"],
    },
    {
        "query": "How do I summon and defeat the Wither?",
        "relevant_titles": ["Wither", "Soul Sand", "Wither Skeleton"],
    },
    # Mobs and combat (61-70)
    {
        "query": "How do I defeat a creeper without getting blown up?",
        "relevant_titles": ["Creeper", "Shield", "Bow"],
    },
    {
        "query": "How do skeletons attack players?",
        "relevant_titles": ["Skeleton", "Bow", "Arrow"],
    },
    {
        "query": "Why do endermen become hostile?",
        "relevant_titles": ["Enderman", "The End"],
    },
    {
        "query": "How do I stop zombies from attacking me?",
        "relevant_titles": ["Zombie", "Light", "Difficulty"],
    },
    {
        "query": "How do I defeat a raid's ravagers?",
        "relevant_titles": ["Ravager", "Raid", "Pillager"],
    },
    {
        "query": "How do I deal with phantoms?",
        "relevant_titles": ["Phantom", "Bed"],
    },
    {
        "query": "Where do slimes spawn?",
        "relevant_titles": ["Slime", "Swamp"],
    },
    {
        "query": "How do I get a trident?",
        "relevant_titles": ["Trident", "Drowned"],
    },
    {
        "query": "How do I use a totem of undying?",
        "relevant_titles": ["Totem of Undying", "Evoker"],
    },
    {
        "query": "How do I summon a beacon?",
        "relevant_titles": ["Beacon", "Wither", "Nether Star"],
    },
    # Redstone and automation (71-80)
    {
        "query": "How does redstone dust transmit signals?",
        "relevant_titles": ["Redstone Dust", "Redstone circuits"],
    },
    {
        "query": "What does a redstone repeater do?",
        "relevant_titles": ["Redstone Repeater", "Redstone circuits"],
    },
    {
        "query": "What does a redstone comparator do?",
        "relevant_titles": ["Redstone Comparator", "Redstone circuits"],
    },
    {
        "query": "How do I build an automatic door?",
        "relevant_titles": ["Redstone", "Piston", "Pressure Plate"],
    },
    {
        "query": "How do sticky pistons work?",
        "relevant_titles": ["Sticky Piston", "Piston", "Redstone circuits"],
    },
    {
        "query": "How does an observer detect changes?",
        "relevant_titles": ["Observer", "Redstone circuits"],
    },
    {
        "query": "What is the difference between a dispenser and a dropper?",
        "relevant_titles": ["Dispenser", "Dropper", "Redstone circuits"],
    },
    {
        "query": "How do I transport items automatically?",
        "relevant_titles": ["Hopper", "Chest", "Minecart with Hopper"],
    },
    {
        "query": "How do I power a redstone lamp?",
        "relevant_titles": ["Redstone Lamp", "Redstone Torch", "Redstone Dust"],
    },
    {
        "query": "How do I create a simple redstone switch?",
        "relevant_titles": ["Lever", "Button", "Redstone circuits"],
    },
    # Storage, transport and exploration (81-90)
    {
        "query": "What does a hopper do?",
        "relevant_titles": ["Hopper", "Chest", "Redstone"],
    },
    {
        "query": "What is an ender chest?",
        "relevant_titles": ["Ender Chest", "Obsidian", "Eye of Ender"],
    },
    {
        "query": "How do shulker boxes help with storage?",
        "relevant_titles": ["Shulker Box", "Shulker", "Inventory"],
    },
    {
        "query": "How do I travel faster using minecarts?",
        "relevant_titles": ["Minecart", "Rail", "Powered Rail"],
    },
    {
        "query": "How do I make a boat?",
        "relevant_titles": ["Boat", "Planks"],
    },
    {
        "query": "How do I find buried treasure?",
        "relevant_titles": ["Buried Treasure", "Explorer Map", "Shipwreck"],
    },
    {
        "query": "How do I find an ocean monument?",
        "relevant_titles": ["Ocean Monument", "Elder Guardian", "Explorer Map"],
    },
    {
        "query": "How do I avoid drowning underwater?",
        "relevant_titles": ["Water Breathing", "Turtle Shell", "Respiration"],
    },
    {
        "query": "How do I fish for items?",
        "relevant_titles": ["Fishing", "Fishing Rod", "Luck of the Sea"],
    },
    {
        "query": "How do I stop losing my items when I die?",
        "relevant_titles": ["Death", "Game rule", "Grave"],
    },
    # Game mechanics and advanced topics (91-100)
    {
        "query": "How does experience work in Minecraft?",
        "relevant_titles": ["Experience", "Enchantment", "Mending"],
    },
    {
        "query": "How do I get more experience quickly?",
        "relevant_titles": ["Experience", "Mob", "Smelting"],
    },
    {
        "query": "How does sleeping affect hostile mobs?",
        "relevant_titles": ["Bed", "Phantom", "Daylight Cycle"],
    },
    {
        "query": "How does light prevent hostile mobs from spawning?",
        "relevant_titles": ["Light", "Mob Spawning", "Torch"],
    },
    {
        "query": "How do I use a name tag on a mob?",
        "relevant_titles": ["Name Tag", "Anvil", "Mob"],
    },
    {
        "query": "How do I tame a wolf?",
        "relevant_titles": ["Wolf", "Bone"],
    },
    {
        "query": "How do I tame a cat?",
        "relevant_titles": ["Cat", "Raw Cod", "Raw Salmon"],
    },
    {
        "query": "How do I ride and control a horse?",
        "relevant_titles": ["Horse", "Saddle", "Horse Armor"],
    },
    {
        "query": "How do I make fireworks for Elytra flight?",
        "relevant_titles": ["Firework Rocket", "Elytra", "Gunpowder"],
    },
    {
        "query": "What happens when I trigger a raid?",
        "relevant_titles": ["Raid", "Bad Omen", "Pillager"],
    },
]


def _rank(matches: list[Match], relevant_titles: list[str]) -> int | None:
    relevant = {title.casefold() for title in relevant_titles}

    for rank, match in enumerate(matches, start=1):
        if match.title.casefold() in relevant:
            return rank

    return None


def _recall_at_k(
    matches: list[Match],
    relevant_titles: list[str],
    k: int,
) -> float:
    relevant = {title.casefold() for title in relevant_titles}
    retrieved = {match.title.casefold() for match in matches[:k]}

    if not relevant:
        return 0.0

    return len(relevant & retrieved) / len(relevant)


def _debug_ranking(
    query: str,
    dense_results: list[Match],
    reranked_results: list[Match],
    relevant_titles: list[str],
    top_n: int = 10,
) -> None:
    relevant = {title.casefold() for title in relevant_titles}

    print(f"\n{'=' * 70}")
    print(f"Query: {query}")
    print(f"Relevant titles: {relevant_titles}")

    for label, matches in (
        ("COSINE RANKING", dense_results),
        ("RERANKED RANKING", reranked_results),
    ):
        print(f"\n{label}")

        for rank, match in enumerate(matches[:top_n], start=1):
            marker = "*" if match.title.casefold() in relevant else " "
            score = "-" if match.rerank_score is None else f"{match.rerank_score:.4f}"

            print(
                f"{rank:>2}. {marker} {match.title:<35} "
                f"cosine={match.cosine:.4f} "
                f"rerank={score}"
            )

            print(f"    Content: {match.content[:700]!r}")


def _metrics(ranks: list[int | None]) -> dict:
    # queries where expected page was found
    scored = [r for r in ranks if r is not None]
    total = len(ranks)
    return {
        # measures the proportion of queries where the expected page appears first
        "hit@1": (sum(1 for r in scored if r <= 1) / total) if total else 0.0,
        # measures the proportion of queries where the expected page appears in the top 5
        "hit@5": (sum(1 for r in scored if r <= 5) / total) if total else 0.0,
        # reward
        "mrr": (sum(1 / r for r in scored) / total) if total else 0.0,
        "found": len(scored),
        "total": total,
    }


def main() -> None:
    database = ChromaStore(str(CHROMA_DIR), COLLECTION_NAME)
    reranker = Reranker(RERANK_MODEL, DEVICE)
    embedder = Embedder(EMBED_MODEL, DEVICE)

    if len(database) == 0:
        print("Index is empty — run `python -m src.cli ingest` first.")
        return

    indexed = database.indexed_titles()
    cosine_ranks: list[int | None] = []
    rerank_ranks: list[int | None] = []
    skipped = 0
    outcomes = Counter()
    dense_recall_1 = []
    dense_recall_5 = []
    rerank_recall_1 = []
    rerank_recall_5 = []

    for case in TEST_CASES:
        query = case["query"]
        relevant_titles = case["relevant_titles"]

        # we dont want to check non-existing sources
        # Skip unindexed title rather than calling it a failure
        indexed_set = {title.casefold() for title in indexed}

        missing_titles = [
            title for title in relevant_titles if title.casefold() not in indexed_set
        ]

        if missing_titles:
            skipped += 1
            print(f"SKIP (relevant titles not in index: {missing_titles}) <- {query}")
            continue

        # shouldn't access private methods directly
        # matches have gone through cosine similarity and reranking
        dense_results = reranker.retrieve(
            query,
            RETRIEVE_TOP_N,
            embedder,
            database,
            QUERY_INSTRUCTION,
        )

        reranked_results = reranker.rerank(query, dense_results.copy())

        if query in DEBUG_QUERIES:
            _debug_ranking(
                query,
                dense_results,
                reranked_results,
                relevant_titles,
            )

        dense_rank = _rank(dense_results, relevant_titles)
        rerank_rank = _rank(reranked_results, relevant_titles)

        if dense_rank is None:
            outcome = "RETRIEVAL MISS"
        elif rerank_rank is None:
            outcome = "RERANK DROP"
        elif rerank_rank < dense_rank:
            outcome = "IMPROVED"
        elif rerank_rank > dense_rank:
            outcome = "REGRESSED"
        else:
            outcome = "UNCHANGED"

        outcomes[outcome] += 1

        cosine_ranks.append(dense_rank)
        rerank_ranks.append(rerank_rank)
        dense_recall_1.append(_recall_at_k(dense_results, relevant_titles, 1))
        dense_recall_5.append(_recall_at_k(dense_results, relevant_titles, 5))

        rerank_recall_1.append(_recall_at_k(reranked_results, relevant_titles, 1))
        rerank_recall_5.append(_recall_at_k(reranked_results, relevant_titles, 5))

        def fmt(rank: int | None) -> str:
            return f"#{rank}" if rank else "miss"

        status = "PASS" if rerank_rank == 1 else ("OK" if rerank_rank else "FAIL")
        print(
            f"{status:<4} {fmt(dense_rank):>5} -> {fmt(rerank_rank):<5} "
            f"{outcome:<4} "
            f"{relevant_titles[0]:<18} <- {query}"
        )

    cosine_metrics = _metrics(cosine_ranks)
    rerank_metrics = _metrics(rerank_ranks)

    if rerank_metrics["total"] == 0:
        print("\nNo test query had its expected page in the index — nothing to score.")
        return

    print(f"\nEvaluated: {rerank_metrics['total']} queries ({skipped} skipped)")
    print(f"{'':24} {'cosine':>8} {'rerank':>8}")
    for key in ("hit@1", "hit@5", "mrr"):
        print(f"{key:24} {cosine_metrics[key]:>8.3f} {rerank_metrics[key]:>8.3f}")

    print("\nReranking outcomes:")
    for outcome in (
        "IMPROVED",
        "REGRESSED",
        "UNCHANGED",
        "RETRIEVAL MISS",
        "RERANK DROP",
    ):
        print(f"{outcome:16} {outcomes[outcome]:>3}")

    print(
        f"{'recall@1':<20}"
        f"{sum(dense_recall_1) / len(dense_recall_1):>8.3f}"
        f"{sum(rerank_recall_1) / len(rerank_recall_1):>8.3f}"
    )
    print(
        f"{'recall@5':<20}"
        f"{sum(dense_recall_5) / len(dense_recall_5):>8.3f}"
        f"{sum(rerank_recall_5) / len(rerank_recall_5):>8.3f}"
    )


if __name__ == "__main__":
    main()
