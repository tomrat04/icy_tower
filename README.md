# Icy Tower — uczenie ze wzmocnieniem

Platformówka 2D w stylu Icy Tower, losową mapą i środowiskiem Gymnasium (PPO).

## Zasady gry

| Element          | Opis                                                                            |
|------------------|---------------------------------------------------------------------------------|
| Sterowanie       | ←/→ lub A/D — ruch, spacja — skok                                               |
| Skok w miejscu   | Niski — około 1 piętro w górę                                                   |
| Rozbieg          | Rozbieg na ziemi buduje pęd pionowy - im większy, tym wyższy skok (do 3 pięter) |
| Ściany           | Blokują ruch w poziomie                                                         |
| Mapa             | Do poziomu 25: szerokie platformy, Od poziomu 26: wąskie platformy              |
| Przegrana        | Wypadnięcie poniżej dolnej krawędzi                                             |
| Wygrana          | Poziom 50                                                                       |

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
- **Akcje:** 6 (brak, lewo, prawo, skok, lewo + skok, prawo + skok)
- **Obserwacje (98 cech):** m.in.: `x`, `y`, `vx`, `vy`, pęd pionowy, margines od dołu ekranu, combo + 12 najbliższych platform (pozycja względem agenta, szerokość)
