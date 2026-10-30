"""Subject-verb agreement prompts with an attractor noun.

Three frames; in all of them the next token should be a verb whose number matches
the subject N1 (the concept Z), not the attractor N2:
  pp:     "The <N1> <prep> the <N2>"                     e.g. "The key near the cabinets"
  orc:    "The <N1> that the <N2> <verb>"                e.g. "The author that the guards like"
  orc_pp: "The <N1> that the <N2> <prep> the <N3> <verb>"
          e.g. "The author that the guards near the doors like"
In the object relative clause the embedded verb agrees with N2, which adds a
second cue for the wrong number, so attraction errors are more common. orc_pp adds
a third cue (N3, same number as N2) and puts more distance between subject and verb.

Every noun form, preposition and verb form is a single token, so prompts within a
frame have the same length and token positions line up across items.
"""

import argparse
import itertools
import random
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

NOUNS = [
    ("key", "keys"), ("book", "books"), ("door", "doors"), ("car", "cars"),
    ("dog", "dogs"), ("cat", "cats"), ("house", "houses"), ("table", "tables"),
    ("chair", "chairs"), ("window", "windows"), ("boy", "boys"), ("girl", "girls"),
    ("friend", "friends"), ("student", "students"), ("teacher", "teachers"),
    ("doctor", "doctors"), ("farmer", "farmers"), ("author", "authors"),
    ("pilot", "pilots"), ("officer", "officers"), ("worker", "workers"),
    ("player", "players"), ("manager", "managers"), ("driver", "drivers"),
    ("painting", "paintings"), ("picture", "pictures"), ("letter", "letters"),
    ("report", "reports"), ("road", "roads"), ("bridge", "bridges"),
    ("tree", "trees"), ("flower", "flowers"), ("bottle", "bottles"),
    ("box", "boxes"), ("bag", "bags"), ("shoe", "shoes"), ("phone", "phones"),
    ("computer", "computers"), ("building", "buildings"), ("room", "rooms"),
    ("school", "schools"), ("river", "rivers"), ("lake", "lakes"),
    ("island", "islands"), ("horse", "horses"), ("bird", "birds"),
    ("ship", "ships"), ("boat", "boats"), ("plane", "planes"), ("train", "trains"),
    ("star", "stars"), ("cup", "cups"), ("plate", "plates"), ("wall", "walls"),
    ("file", "files"), ("document", "documents"), ("sign", "signs"),
    ("song", "songs"), ("game", "games"), ("machine", "machines"),
    ("engine", "engines"), ("cabinet", "cabinets"), ("desk", "desks"),
    ("lamp", "lamps"), ("lawyer", "lawyers"),
    ("soldier", "soldiers"), ("singer", "singers"), ("nurse", "nurses"),
    ("king", "kings"), ("village", "villages"), ("garden", "gardens"),
]
# Subjects of the embedded verb in the orc frame.
ANIMATE = {
    "boy", "girl", "friend", "student", "teacher", "doctor", "farmer", "author",
    "pilot", "officer", "worker", "player", "manager", "driver", "dog", "cat",
    "horse", "bird", "lawyer", "soldier", "singer", "nurse", "king",
}
PREPS = ["of", "near", "behind", "beside", "on", "under"]
# (form agreeing with a singular subject, form agreeing with a plural subject)
VERBS = [
    ("likes", "like"), ("sees", "see"), ("loves", "love"), ("hates", "hate"),
    ("knows", "know"), ("helps", "help"), ("meets", "meet"), ("calls", "call"),
    ("needs", "need"), ("follows", "follow"), ("visits", "visit"),
    ("trusts", "trust"), ("watches", "watch"), ("hears", "hear"),
    ("remembers", "remember"), ("finds", "find"), ("wants", "want"),
]
FRAMES = ("pp", "orc", "orc_pp")


def is_single_token(tokenizer, word):
    return len(tokenizer.encode(" " + word)) == 1


def sample_orc_pp(rng, nouns, animate, preps, verbs, n):
    """Draw n distinct (N1, N2, N3, prep, verb) combos; too many to enumerate."""
    combos, seen = [], set()
    while len(combos) < n:
        combo = (rng.choice(nouns), rng.choice(animate), rng.choice(nouns),
                 rng.choice(preps), rng.choice(verbs))
        if len(set(combo[:3])) == 3 and combo not in seen:
            seen.add(combo)
            combos.append(combo)
    return combos


def build(tokenizer_name, n_per_cell, seed, frames=FRAMES):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    nouns = sorted({
        pair for pair in NOUNS if all(is_single_token(tokenizer, w) for w in pair)
    })
    animate = [pair for pair in nouns if pair[0] in ANIMATE]
    preps = [p for p in PREPS if is_single_token(tokenizer, p)]
    verbs = [pair for pair in VERBS if all(is_single_token(tokenizer, w) for w in pair)]

    rng = random.Random(seed)
    rows = []
    for frame in frames:
        if frame == "pp":
            combos = [(n1, n2, p) for n1, n2 in itertools.permutations(nouns, 2) for p in preps]
        elif frame == "orc":
            combos = [(n1, n2, v) for n1 in nouns for n2 in animate if n1 != n2 for v in verbs]
        # Balanced 2x2: subject number (z) x attractor number.
        for z, a in itertools.product([0, 1], repeat=2):
            if frame == "orc_pp":
                cell = sample_orc_pp(rng, nouns, animate, preps, verbs, n_per_cell)
            else:
                cell = rng.sample(combos, n_per_cell)
            for combo in cell:
                n1, n2 = combo[:2]
                row = {"frame": frame, "subject": n1[0], "attractor": n2[0], "z": z, "attr_plural": a}
                if frame == "pp":
                    row["prompt"] = f"The {n1[z]} {combo[2]} the {n2[a]}"
                    row["link"] = combo[2]
                elif frame == "orc":
                    row["prompt"] = f"The {n1[z]} that the {n2[a]} {combo[2][a]}"
                    row["link"] = combo[2][1]
                else:
                    n3, prep, verb = combo[2:]
                    row["prompt"] = f"The {n1[z]} that the {n2[a]} {prep} the {n3[a]} {verb[a]}"
                    row["link"] = verb[1]
                    row["attractor2"] = n3[0]
                rows.append(row)
    print(f"{len(nouns)} noun pairs ({len(animate)} animate), {len(preps)} prepositions, {len(verbs)} verbs")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="EleutherAI/pythia-160m")
    parser.add_argument("--n-per-cell", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", nargs="+", default=list(FRAMES), choices=FRAMES)
    parser.add_argument("--out", default="results/stimuli.csv")
    args = parser.parse_args()

    df = build(args.tokenizer, args.n_per_cell, args.seed, args.frames)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} prompts")
    print(df.groupby("frame").sample(3, random_state=0).to_string(index=False))


if __name__ == "__main__":
    main()
