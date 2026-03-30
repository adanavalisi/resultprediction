# resultprediction

Bu repo, Transfermarkt verilerini kullanarak futbol mac sonucu icin `1-X-2` tahmin modeli ureten temel bir Python boru hatti sunar.

Akis:

1. `scripts/scrape_transfermarkt.py`
   Transfermarkt uzerinden 9 lig ve son 5 sezon icin mac sonuclari ile takim baglam verilerini toplar.
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
python scripts/scrape_transfermarkt.py --output-dir data/raw --seasons 5
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
- Takim kadro degeri
- Ilk 5 en degerli oyuncu
- Sakat oyuncu listesi
- Stadyum kapasitesi

Not:

- Transfermarkt HTML yapisi degisebildigi icin seciciler zaman zaman guncellenebilir.
- Site rate-limit veya bot korumasi uygulayabilir. Gerekirse `--delay 2.0` gibi daha yuksek gecikme kullanin.
- Tarihsel "ilk 11" bilgisini Transfermarkt mac raporlarindan cekmek isterseniz scraping modulu bu yonde genisletilmeye uygundur, ancak ilk surum mevcut haliyle kadro degeri, ust duzey oyuncu ve sakatlik etkisini takim/sezon baglaminda toplar.

Yeni dogrulama/fail kosullari:

- `scrape_matches` ve `scrape_team_context`, lig+sezon bazinda tablo veya satir sorunlarini `selector_miss`, `blocked`, `no_data` reason code'lariyla `scrape_metadata.json` icindeki `errors[]` alanina yazar.
- Lig+sezon esik kontrolleri uygulanir (`matches >= 100`, `team_context >= 10`). Esik alti durumlar artik sadece stdout'a yazilmaz; `errors[]` icine `threshold_breach` olarak kaydedilir.
- Her lig+sezon icin satir sayilari `rows_per_league_season` alaninda tutulur.
- Betik sonunda `errors[]` bos degilse islem `non-zero exit code` ile biter. Bu durum CI veya otomasyon tarafinda "veri kalitesi/fetch sorunu" olarak yorumlanmalidir.
- `warnings[]` alani bilgilendirici ama kritik olmayan durumlar icin ayrilmistir (su an bos gelebilir).

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
- Bu repo bir ilk surum iskeletidir. Transfermarkt sayfa yapisina gore secicilerde ince ayar gerekebilir.
