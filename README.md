# Icy Tower — uczenie ze wzmocnieniem (Pygame)

Platformówka 2D w stylu **Icy Tower** z proceduralną mapą i środowiskiem **Gymnasium** (DQN).

## Zasady gry

| Element | Opis |
|--------|------|
| Sterowanie | **←/→** lub **A/D** — ruch, **Spacja** — skok |
| Skok w miejscu | Niski — ok. **1 piętro** w górę |
| Rozbieg | ←/→ na ziemi buduje **pęd pionowy** (pasek); im pełniejszy, tym wyższy skok (do **2 pięter**) |
| Wall jump | Przy ścianie + **Spacja** — odbicie w bok i **boost w górę** (`vy`), nie szybszy bieg w poziomie |
| Combo | Jednym skokiem min. **2 piętra** w górę (np. 10→12) — startuje **pasek timera** |
| Utrzymanie combo | Zanim timer zniknie, kolejny skok o ≥2 piętra — dłuższa seria = więcej punktów na koniec |
| Mapa | Do poz. 100: 1 szeroka platforma; od 100: 75% × 1 wąska, 25% × 2 wąskie |
| Przegrana | Wypadnięcie poniżej dolnej krawędzi ekranu (auto-scroll) |
| Wygrana | Poziom **500** |

## Instalacja

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Uruchomienie

```bash
python play.py
```

Trening: `python train.py` (domyślnie **1 000 000** kroków, 8 równoległych envów)

Test agenta: `python test.py --model models/best/best_model`

## Uczenie ze wzmocnieniem

- **Akcje:** 0–5 (ruch, skok, kombinacje)
- **Obserwacja (102 cechy):** `x`, `y`, `vx`, `vy`, pęd pionowy, margines od dołu ekranu, combo + **12 najbliższych platform** (pozycja względem agenta, szerokość, szansa lądowania)
- Stary model wymaga **ponownego treningu**

Logika: `icy_tower/game.py`, `icy_tower/config.py`
