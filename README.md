# resultprediction

Bu repo, API-Football verilerini kullanarak futbol mac sonucu icin `1-X-2` tahmin modeli ureten temel bir Python boru hatti sunar.

Akis:

1. `scripts/scrape_transfermarkt.py`
   API-Football uzerinden 9 lig ve son 5 sezon icin mac sonuclari ile takim baglam verilerini toplar.
2. `scripts/prepare_dataset.py`
   Ham veriyi ozelliklere donusturur ve egitim veri setini uretir.
3. `scripts/train_dnn.py`
   DNN modeli egitir ve ev sahibi galibiyeti, beraberlik, deplasman galibiyeti olasiliklarini verir.
4. `app.py`
   Streamlit tabanli web arayuzu ile secilen iki takim icin tahmin olasiliklarini grafik halinde gosterir.

Desteklenen ligler:

- Turkiye Super Lig
- Ingiltere Premier League
- Almanya Bundesliga
- Ispanya LaLiga
- Italya Serie A
- Fransa Ligue 1
- Portekiz Liga Portugal
- Belcika Jupiler Pro League
- Hollanda Eredivisie

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Web arayuzu

Model ve veri artefaktlari hazir olduktan sonra arayuzu su komut ile acabilirsiniz:

```bash
streamlit run app.py
```

Arayuz ozellikleri:

- Koyu gri arka plan uzerinde merkezde buyuk beyaz `VS`
- Sol ve sag tarafta lig ve takim secim kutulari
- Ortadaki buton ile secilen eslesme icin tahmin alma
- Asagida `Home Win`, `Draw`, `Away Win` olasiliklarini grafik ve yuzde kartlari ile gosterme

On kosul:

- `data/raw/matches.csv`
- `data/raw/team_context.csv`
- `data/processed/training_dataset.parquet`
- `data/processed/feature_columns.json`
- `models/football_outcome_dnn.keras`
- `models/feature_scaler.joblib`

## 1. Veri cekme

```bash
python scripts/scrape_transfermarkt.py --output-dir data/raw --seasons 5 --api-key <API_FOOTBALL_KEY>
```

Uretilen dosyalar:

- `data/raw/matches.csv`
- `data/raw/team_context.csv`
- `data/raw/scrape_metadata.json`

Toplanan alanlar:

- Lig, sezon, mac tarihi
- Ev sahibi ve deplasman takimlari
- Gol sayilari
- Seyirci sayisi ve stadyum bilgisi
- Takim stadyum kapasitesi
- API kapsaminda mevcut oldugu kadar seyirci ve stadyum bilgisi
- API kaynakli eksik alanlar icin guvenli bos degerler

Not:

- API anahtari zorunludur. `--api-key` ile verebilir veya `API_FOOTBALL_KEY` environment variable'ina koyabilirsiniz.
- API rate-limit uyguladigi icin gecikme ve timeout degerlerini `--delay` ve `--timeout` ile ayarlayabilirsiniz.
- `scrape_metadata.json` dosyasi, lig/sezon bazli satir sayilari ile API hatalarini `errors[]` alaninda raporlar.
- API-Football, Transfermarkt'taki piyasa degeri/oyuncu degeri gibi alanlari dogrudan saglamadigi icin bu alanlar bos degerle yazilir.

## 2. Veri hazirlama

```bash
python scripts/prepare_dataset.py --raw-dir data/raw --output-dir data/processed
```

Uretilen ozellikler:

- Son 5 mac agirlikli form
- Son 10 mac form ortalamasi
- Son 5 mac gol atma ve yeme ortalamalari
- Son 10 mac gol farki
- Son 5 mac galibiyet sayisi
- Ikili rekabette son 5 mac puan ve gol farki
- Kadro degeri farki
- Kilit oyuncu toplam degeri farki
- Sakat kilit oyuncu farki
- Seyirci / stadyum kapasitesi orani
- Ev sahibi etkisi

Uretilen dosyalar:

- `data/processed/training_dataset.parquet`
- `data/processed/feature_columns.json`

## 3. Model egitimi

```bash
python scripts/train_dnn.py --data-path data/processed/training_dataset.parquet
```

Kaydedilen ciktilar:

- `models/football_outcome_dnn.keras`
- `models/feature_scaler.joblib`
- `models/metrics.json`
- `models/history.json`
- `models/test_predictions.csv`

`test_predictions.csv` dosyasinda her mac icin su kolonlar yer alir:

- `home_win_pct`
- `draw_pct`
- `away_win_pct`
- `predicted_label`

`predicted_label` degerleri:

- `1`: Ev sahibi galibiyeti
- `X`: Beraberlik
- `2`: Deplasman galibiyeti

## Teknik notlar

- Zaman sizmasini azaltmak icin egitim / dogrulama / test ayrimi sezon bazli yapilir.
- Model, `softmax` cikis katmani ile yuzdelik olasilik uretir.
- Bu repo bir ilk surum iskeletidir. API kapsami ve plan limitlerine gore veri derinligi degisebilir.
