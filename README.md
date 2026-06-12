# Icy Tower — uczenie ze wzmocnieniem

Platformówka 2D w stylu **Icy Tower** z proceduralną mapą i środowiskiem **Gymnasium** (PPO).

## Zasady gry

| Element | Opis |
|--------|------|
| Sterowanie | **←/→** lub **A/D** — ruch, **Spacja** — skok |
| Skok w miejscu | Niski — ok. **1 piętro** w górę |
| Rozbieg | ←/→ na ziemi buduje **pęd pionowy** (pasek); im pełniejszy, tym wyższy skok (do **2 pięter**) |
| Ściany | Blokują ruch w poziomie — bez odbicia |
| Combo | Jednym skokiem min. **2 piętra** w górę (np. 10→12) — startuje **pasek timera** |
| Utrzymanie combo | Zanim timer zniknie, kolejny skok o ≥2 piętra — dłuższa seria = więcej punktów na koniec |
| Mapa | Do poz. 100: 1 szeroka platforma; od 100: 75% × 1 wąska, 25% × 2 wąskie |
| Przegrana | Wypadnięcie poniżej dolnej krawędzi ekranu|
| Wygrana | Poziom **500** |

## Uruchomienie gry

```bash
python play.py
```

## Trening

```bash
python train.py
```

## Test agenta

```bash
python test.py 
```

## Uczenie ze wzmocnieniem

- **Algorytm:** PPO
- **Akcje:** 0–5 (lewo, prawo, skok - różne kombinacje)
- **Obserwacje (102 cechy):** `x`, `y`, `vx`, `vy`, pęd pionowy, margines od dołu ekranu, combo + **12 najbliższych platform** (pozycja względem agenta, szerokość, szansa lądowania)
