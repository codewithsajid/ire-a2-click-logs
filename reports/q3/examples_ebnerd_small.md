# Lexical vs semantic — worked examples (ebnerd/small, test)

Query = terms/vectors of each user's last 30 clicked articles. Universe = 1,677 articles live in the 7 days around the split. `*` marks an article the user actually clicked.

## user 8807 — 959 clicks in history

**History (most recent 6):**
- [krimi] Barn mistænkt for at stå bag tragisk dødsfald
- [nyheder] Trist dansk måling: Trivslen daler i folkeskolerne
- [krimi] Dansk ægtepar anholdt på græsk ferieø
- [sport] Skandaløse scener: Stjerne smidt ud efter raserianfald
- [nyheder] Multimillionær stoppet i vildt flugtforsøg
- [krimi] Sandberg stemplet for livet: 'Bliver nødt til at smadre dig'

| # | BM25 | emb |
|---|------|-----|
| 1 | [krimi] To mænd anholdt for bortførelse af 10-årig pige | [krimi] Raser over politiet: Brugte elektrochokvåben mo… |
| 2 | *[krimi] Mystisk ulykke: To børn dræbt og mand anholdt | [krimi] Knivoffer og rockerpræsident: Han styrer 'Aalbo… |
| 3 | *[krimi] Buler og blod: Politiet fandt ham på jorden | [nyheder] USA: Wagner-gruppen vil smugle våben |
| 4 | [krimi] 26-årig fængslet for drabsforsøg - sigtet for a… | [krimi] Et år efter skoleskyderi: Her købte gerningsman… |
| 5 | [krimi] Voldtægtsforsøg af 25-årig: Politi jagter mand … | [nyheder] Mand i kassevogn ville angiveligt dræbe Biden o… |

top-10 overlap between the two retrievers: **0/10**

## user 16739 — 253 clicks in history

**History (most recent 6):**
- [underholdning] Gennemsigtig kjole på den røde løber: Derfor gør hun det
- [nyheder] Elafgift rammer 2,8 millioner danskere
- [nyheder] Trump gør grin med konkurrentens tekniske problemer
- [nyheder] DMI: Forårssol til det meste af landet
- [nationen] Returamok: 42.000 Boozt-kunder ramt
- [musik] Vild hyre: Vælter sig i millioner

| # | BM25 | emb |
|---|------|-----|
| 1 | [nyheder] 22 grader og torden: Her rammer det | [krimi] Raser over politiet: Brugte elektrochokvåben mo… |
| 2 | [krimi] To mænd anholdt for bortførelse af 10-årig pige | [underholdning] Med døden i nakken |
| 3 | [underholdning] Christiane: Jeg fik en taknemmelighed for livet | *[nyheder] Mor fik chok: Så forsvundet skuespiller-søn i R… |
| 4 | [krimi] Skuddrama på restaurant: Anholdt to gange | [nyheder] USA: Wagner-gruppen vil smugle våben |
| 5 | [krimi] Mystisk ulykke: To børn dræbt og mand anholdt | *[nyheder] Tre dages McCann-efterforskning fører intet gen… |

top-10 overlap between the two retrievers: **0/10**

## user 4289 — 297 clicks in history

**History (most recent 6):**
- [forbrug] Den falske telefon-myte: Du skal ikke gøre det
- [krimi] Person stukket med kniv
- [nyheder] Trist dansk måling: Trivslen daler i folkeskolerne
- [krimi] Dansk ægtepar anholdt på græsk ferieø
- [nyheder] En ældre dronning skal pryde nye mønter
- [krimi] Motorvej spærret: Ild i køretøj

| # | BM25 | emb |
|---|------|-----|
| 1 | *[krimi] Mystisk ulykke: To børn dræbt og mand anholdt | [nationen] Claus til politiet: Giv mig min kniv |
| 2 | [krimi] Person stukket med kniv | [krimi] Raser over politiet: Brugte elektrochokvåben mo… |
| 3 | [krimi] Hollywood: Ni personer skudt - herunder 1-årig | [krimi] Politi: Efterforsker påkørsel som drabsforsøg |
| 4 | [krimi] Politi har fundet efterlyst 12-årig | [nationen] Ond stemning på minigolf |
| 5 | *[krimi] To mænd anholdt for bortførelse af 10-årig pige | [nyheder] 95-årig australsk oldemor er død efter strømsku… |

top-10 overlap between the two retrievers: **0/10**

## Overall

- mean top-10 overlap between BM25 and embeddings: **0.55/10** — the two retrieve largely different articles, so their errors are not the same errors
- BM25 query terms per user (median): 777