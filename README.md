# Icy Tower — uczenie ze wzmocnieniem (Pygame)

Platformówka 2D w stylu **Jump King** / Icy Tower z proceduralną mapą i środowiskiem **Gymnasium** do treningu agenta (DQN).

## Zasady gry

| Element | Opis |
|--------|------|
| Sterowanie | **A** — lewo, **D** — prawo |
| Wspinaczka | Automatyczne odbicie po lądowaniu; wyższy pierwszy skok na starcie |
| Platformy jednokierunkowe | W locie w górę przelatujesz przez platformy; przy spadaniu lądujesz na wierzchu |
| Mapa | Równe odstępy pionowe; na każdym poziomie **1–3** platformy o losowej długości |
| Przegrana | **3.** lądowanie na tej samej platformie (max **2** odbicia) **lub** spadek wyraźnie poniżej ostatniej platformy |
| Wygrana | Osiągnięcie **poziomu 500** (wysokość na mapie, nie liczba odbić) |

## Instalacja

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Uruchomienie

**Gra ręczna:**

```bash
python play.py
```

- **R** — restart po śmierci  
- **ESC** — wyjście  

**Trening DQN:**

```bash
python train.py --timesteps 200000 --n-envs 4
```

**TensorBoard** (w drugim terminalu):

```bash
tensorboard --logdir logs/tensorboard
```

Logi SB3: nagroda, loss, epsilon, ewaluacja. Metryka gry: `game/highest_level` (najwyższy poziom w epizodzie).

Opcje: `--tb-log logs/tensorboard`, `--run-name moj_eksperyment`

**Test agenta** (60 FPS, jak `play.py`):

```bash
python test.py --model models/best/best_model
```

Opcje: `--fps 60`, `--pause-on-end 2`, **R** — restart, **ESC** — wyjście

## Struktura projektu

```
icy_tower/
  config.py      # stałe fizyki i mapy
  entities.py    # gracz, platforma
  world.py       # generator poziomów
  game.py        # logika i reguły
  render.py      # rysowanie Pygame
  env.py         # Gymnasium Env
play.py          # gra człowiek
train.py         # trening SB3 (+ pasek postępu tqdm)
test.py          # podgląd modelu w normalnym tempie
```

## Uczenie ze wzmocnieniem

- **Akcje:** `0` — bez ruchu, `1` — lewo, `2` — prawo  
- **Nagroda:** +1 za nowy poziom platformy, kara przy śmierci, bonus przy wygranej  
- **Obserwacja (46 cech):** gracz + **10 poziomów** platform nad ostatnią platformą odbicia (x, y, szerokość, level)  
- Stary model (18 cech) nie zadziała — wytrenuj od nowa po zmianie obserwacji  

Logika obserwacji: `icy_tower/observations.py`

## Dalsze kroki

- Curriculum learning (mniej platform na start)  
- Nagroda kształtująca za zbliżanie się do najbliższej platformy  
- Frame stacking / CNN jeśli dodasz render jako obraz  
