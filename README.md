# resultprediction

Bu repo, openfootball tarihsel veri seti ile modeli egitip API-Football ile gunluk/anlik fikstur tahmini ureten bir Python boru hatti sunar.

Akis:

1. `scripts/ingest_openfootball.py`
   openfootball/football.json veri tabanindan tarihsel mac verilerini `data/raw` altina indirir.
2. `scripts/prepare_dataset.py`
   Ham veriyi ozelliklere donusturur ve egitim veri setini uretir.
3. `scripts/train_dnn.py`
   DNN modeli egitir ve ev sahibi galibiyeti, beraberlik, deplasman galibiyeti olasiliklarini verir.
4. `scripts/predict_live_fixtures.py`
   API-Football'dan belirli bir tarihteki fiksturleri alip egitilmis modelle anlik tahmin uretir.
5. `app.py`
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

## 1. Egitim icin openfootball veri cekme

```bash
python scripts/ingest_openfootball.py --output-dir data/raw --seasons 8
```

Uretilen dosyalar:

- `data/raw/matches.csv`
- `data/raw/team_context.csv`
- `data/raw/openfootball_metadata.json`

Toplanan alanlar:

- Lig, sezon, mac tarihi
- Ev sahibi ve deplasman takimlari
- Gol sayilari
- Seyirci sayisi ve stadyum bilgisi
- Takim stadyum kapasitesi
- openfootball kapsaminda mevcut oldugu kadar mac sonucu ve tarih bilgisi
- Eksik baglam alanlari (seyirci, stadyum, piyasa degeri) icin guvenli bos degerler

Not:

- openfootball cekiminde API anahtari gerekmez.
- `openfootball_metadata.json` dosyasi, indirilen URL'leri ve hata kayitlarini `errors[]` alaninda raporlar.
- openfootball veri setinde piyasa degeri/sakat oyuncu alanlari olmadigi icin bu alanlar model uyumlulugu adina bos degerlerle uretilir.

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


## 4. API-Football ile gunluk/anlik fikstur tahmini

Model egitildikten sonra secilen lig ve tarihteki fiksturler icin tahmin alabilirsiniz:

```bash
python scripts/predict_live_fixtures.py --league "Premier League" --date 2026-04-06 --api-key <API_FOOTBALL_KEY>
```

Cikti: `data/predictions/live_predictions.json`

Notlar:
- Bu adim API anahtari gerektirir (`--api-key` veya `API_FOOTBALL_KEY`).
- API'den gelen takim isimleri egitim verisindeki takim isimleriyle birebir eslesmelidir.

## Teknik notlar

- Zaman sizmasini azaltmak icin egitim / dogrulama / test ayrimi sezon bazli yapilir.
- Model, `softmax` cikis katmani ile yuzdelik olasilik uretir.
- Bu repo bir ilk surum iskeletidir. API kapsami ve plan limitlerine gore veri derinligi degisebilir.
