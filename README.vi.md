**Đọc bằng ngôn ngữ khác:** [English](README.md) | [Tiếng Việt](README.vi.md)

# CROSS-SECTIONAL FACTOR INVESTING FOR CRYPTO MARKET

Đây là một báo cáo nghiên cứu việc đầu tư dựa vào các yếu tố giữa các tài sản cho thị trường Crypto- Xây dựng một chiến lược long-short dựa trên việc xếp hạng các đồng coin theo các đặc trưng (factors) định lượng thay vì dự đoán giá tuyệt đối

Mục tiêu cuối của dự án là trình bày được toàn bộ quy trình từ khâu chọn universe → tạo factor → kiểm định thống kê → xây alpha → backtest có thực hiện đúng kỷ luật của một quy trình nghiên cứu định lượng chuyên nghiệp hay không: tách bạch giai đoạn khám phá khỏi giai đoạn kiểm định ngoài mẫu (out-of-sample), tránh look-ahead bias, và định lượng được chi phí giao dịch thực tế.

Pipeline gồm 4 notebook (tương ứng với 4 file step trong src) thực hiện tuần tự và phụ thuộc dữ liệu lẫn nhau:

[1] Universe + Raw Factors  →  [2] Statistical Validation  →  [3] Composite Alpha  →  [4] Portfolio & Backtest

## Cấu Trúc dự án

```
crypto-quant-alpha/
├── notebook_analysis/    # 4 notebooks: universe, validation, alpha, backtest
├── src/                  # Reusable modules (features, validation, alpha, backtest)
├── data/                 # Raw OHLCV + processed parquet outputs
├── run_pipeline.py       # Entry point chạy cả 4 bước
├── requirements.txt
└── README.md
```

|  | Notebook và file tương ứng                       | Giai đoạn             | Input                                                  | Output                                                                                           |
| - | ------------------------------ | --------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| 1 | `1_universe_factor_construction.ipynb` (`step1_feature.py`)  | Universe & Raw Factor | Dữ liệu thị trường gốc                                 | Universe & raw factors sạch, point-in-time, chia thành 2020–2021 (In Sample) và 2022–2025 (Out of Sample) |
| 2 | `2_statistical_significant_factor.ipynb` (`step2_validation.py`) | Statistical Filtering | Raw factors — **2020–2021 (IS)**                       | Các factor có bằng chứng thống kê; không sử dụng dữ liệu 2022–2025                           |
| 3 | `3_composite_alpha_construction.ipynb` (`step3_alpha.py`)      | Weighting Scheme      | Các factor đã lọc + dữ liệu **2022–2025**              | Phương pháp weighting bằng walk-forward: Tối ưu trọng số trên quá khứ- áp dụng cho tương lai- dùng cửa sổ trượt về phía trước và lặp lại để kết hợp 2 factor |
| 4 | `4_backtesting.ipynb` (`step4_backtest.py`)       | Execution & Cost      | Alpha + weighting scheme + dữ liệu **2022–2025 (OOS)** | Danh mục thực thi, transaction costs, turnover và hiệu suất OOS                           |


### Kết quả chính

| Metric           |    Value |
| ---------------- | -------: |
| `n_days`         |     1096 |
| `mean_daily_ret` | 0.001910 |
| `total_return`   | 4.482628 |
| `ann_return`     | 0.617407 |
| `ann_vol`        | 0.423536 |
| `sharpe`         | 1.457746 |
| `sortino`        | 2.163329 |
| `max_drawdown`   | 0.381515 |
| `win_rate`       | 0.563869 |
| `profit_factor`  | 1.209725 |
| `nw_t_stat`      | 2.542229 |
| `p_value`        | 0.011015 |

Một vài nhận xét chính:
  - Kết quả backtest trên 1.096 ngày cho thấy chiến lược có hiệu suất tương đối tốt nhưng đi kèm mức rủi ro đáng kể. Strategy đạt total return khoảng 448,3%, annualized return 61,7%, trong khi annualized volatility ở mức 42,4%. Từ đó, Sharpe ratio đạt 1,46 và Sortino ratio đạt 2,16
  - Rủi ro lớn nhất thể hiện ở maximum drawdown 38,15%. Mặc dù annualized return đạt 61,7%, portfolio từng mất khoảng 38% từ đỉnh xuống đáy trước khi phục hồi. Sau một drawdown 38,15%, portfolio cần tăng khoảng 61,7% để quay trở lại mức vốn ban đầu. Do đó, Sharpe 1,46 không nên được xem xét độc lập với drawdown.

<Figure size 640x480 with 1 Axes><img width="630" height="469" alt="image" src="https://github.com/user-attachments/assets/5366dc90-152f-4b8b-b9bf-ef72b6a436bb" />


Nhận xét từ biểu đồ cửa số trượt Sharp 12 tháng:

  - ~Oct 2023 – Oct 2024 (~12 tháng): rolling Sharpe dao động thấp, phần lớn dưới 1.0, có 2 đáy rõ rệt — một quanh giữa 2024 (~0.6) và một đáy sâu nhất toàn chuỗi vào khoảng Oct 2024 (~0.2). 
  - ~Nov 2024 – Jan 2026 (~14 tháng): Sharpe bật tăng dứt khoát, đạt đỉnh ~2.7 quanh tháng 3/2025, sau đó dao động ổn định trong biên 1.9–2.6 cho đến hết mẫu.
 
 <Figure size 1200x500 with 2 Axes><img width="1102" height="490" alt="image" src="https://github.com/user-attachments/assets/fc0152da-c148-4248-a59e-69265f2145bd" />


Từ đó cho ta thấy rằng đây không phải một alpha ổn định trải đều theo thời gian mà là một chiến lược có hiêu quả mạnh theo thị trường, với việc gần như toàn bộ giá trị được tạo ra trong khoảng 14 thấng cuối. Giá trị returns qua từng tháng biến động lớn, những lần drawdowns đều lớn hơn hẳn so với lần returns dương. Đồng thời kết quả cũng khớp và lí giải tại sao chỉ có 2 factor liên quan tới volatility sống sót sau các kiểm định thống kê.

**Kết luận cuối:** khả năng rất cao chiến lược này đang hưởng lợi từ biến cố thị 
trường cụ thể chứ không phải chiến lược bền vững lâu dài (một phần vì phần 
kiểm định thống kê chỉ từ đầu 2020 tới cuối 2021 nên dữ liệu chưa đủ dài). 

### Cách chạy

#### 1. Clone repo

```bash
git clone https://github.com/HaiNam19/Statistical-Factor-and-Machine-Learning-for-Crypto-Cross-Sectional-Alpha.git
cd Statistical-Factor-and-Machine-Learning-for-Crypto-Cross-Sectional-Alpha
```

#### 2. Tạo virtual environment

```bash
python -m venv .venv
```

#### 3. Kích hoạt virtual environment

Windows (CMD):

```cmd
.venv\Scripts\activate.bat
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

#### 4. Cài dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

Kiểm tra cài đặt thành công:

```bash
python -c "from src.step1_features import Step1Pipeline; print('OK')"
```

Kết quả mong đợi:

```text
OK
```

#### 5. Chuẩn bị dữ liệu và chạy

Đặt file raw `ohlcv_daily.parquet` vào thư mục `data/universe/`.

Kiểm tra file đã đúng vị trí (Windows):

```cmd
dir data\universe\ohlcv_daily.parquet
```

Chạy toàn bộ pipeline:

```bash
python run_pipeline.py
```

Hoặc chạy từng bước:

```bash
python run_pipeline.py --steps 1
python run_pipeline.py --steps 2
python run_pipeline.py --steps 3
python run_pipeline.py --steps 4
```

Các file output được lưu vào `data/processed/`:

```text
B1_factor_construction.parquet
2022-2025_factor_construction.parquet
eligible_factors.parquet
D1_df_weightt.parquet
backtest_plot.png
```

## Mô Tả Từng Phần

### I. 1_universe_factor_construction — Universe & Factor Construction

Mục tiêu: Xây dựng investment universe theo từng ngày (point-in-time) và tạo các raw factors cho từng coin thuộc universe tại thời điểm đó.

Input: ohlcv_daily.parquet — dữ liệu OHLCV daily từ 2020-01-01 đến 2026-08-17.


Universe Construction:
  - Yêu cầu tối thiểu 6 tháng lịch sử giao dịch tại thời điểm t.
  - Loại bỏ stablecoins, leveraged/wrapped tokens và các token không hợp lệ.
  - Chọn Top <=40 theo median_dollar_volume_30d, với thanh khoản tối thiểu $10M/ngày.

Raw Factors — 4 nhóm:
  - Momentum: 7D / 14D / 30D / 90D
  - Reversal: 1D / 3D
  - Volatility: 7D / 14D / 30D + Vol-of-Vol 14D
  - Liquidity: Amihud 14D / 30D
    
Factors được winsorize (5–95%) và cross-sectional z-score theo ngày trong universe. Forward returns được tính cho 1D / 5D / 14D.
Data Split:
  - 2020–2021: Factor discovery → Notebook 2
  - 2022–2025: Weight fitting & OOS evaluation → Notebooks 3–4


### II. 2_statistical_significant_factor.ipynb — Statistical Validation

Mục tiêu: Kiểm định 12 raw factors và chỉ giữ lại các factor có bằng chứng thống kê và tín hiệu kinh tế.

Input: B1_factor_construction.parquet — giai đoạn 2020–2021

Statistical Tests:
  - Daily Spearman IC: Factor vs. forward return, kiểm định mean IC bằng Newey-West HAC.
  - Univariate Fama-MacBeth: Cross-sectional factor return và NW t-stat.
  - Tertile Portfolio Spread: Mua nhóm factor cao, bán nhóm factor thấp, đo spread theo bps/day.

Factor selection rules:
  - Có ít nhất 1/3 tests có ý nghĩa ở mức 5%.
  - C / beta / spread có dấu nhất quán.
  - Spread ≥ 10 bps/day để có khả năng bù transaction costs.
  - Kiểm tra cross-factor correlation và loại bỏ factor dư thừa.

Kết quả:
  - 2/12 factors được giữ lại: z_vol_14d và z_vol_of_vol_14d.
  - Momentum, reversal và Amihud illiquidity đều bị loại.
  - Cả 2 factor được chọn đều thuộc nhóm volatility.

**Limitation:** Giai đoạn 2020–2021 chỉ khoảng 15 tháng và mang đặc trưng của một market regime riêng biệt (COVID), nên kết quả factor selection có thể chưa đại diện cho các market regimes khác. Tôi sẽ có những cải thiện vào các phiên bản sau, tập trung vào đánh giá từng giai đoạn hơn.

### III. 3_composite_alpha_construction.ipynb — Composite Alpha Construction

Mục tiêu: Kết hợp 2 factor đã được chọn (z_vol_14d, z_vol_of_vol_14d) thành một composite alpha score và so sánh các phương pháp weighting.

Input: 2022-2025_factor_construction.parquet — dữ liệu hoàn toàn tách biệt khỏi giai đoạn factor selection ở Notebook 2.

Phương pháp:
  - Walk-forward CV: Rolling 12-month train → 1-month test, hàng tháng từ 2023-01 đến 2025-12.
  - So sánh 4 phương pháp:
      - Equal-weight: Trung bình cộng 2 factor z-score.
      - Evidence-weight: Trọng số theo ic_mean trên từng training window.
      - Ridge Regression: Linear model với regularization để dự đoán forward return.
      - LightGBM: Gradient boosting tương tự Ridge để dự đoán forward return
  - Kết quả:
      - Equal-weight là phương pháp được chọn làm trọng số chính.
      - Ridge và LightGBM không vượt trội hơn equal-weight.
      - Evidence-weight trung bình vượt trội hơn equal-weight nhưng không có ý nghĩa thống kê.
        
**Quyết định:** Chỉ giữ equal-weight nhằm tránh overfitting và đảm bảo tính nhất quán giữa alpha construction và downstream backtest.

### IV. 4_backtesting.ipynb — Backtest

Mục tiêu: Chuyển composite alpha (equal-weight) thành danh mục long/short có thể thực thi, mô phỏng transaction costs và đánh giá hiệu suất.

Input: 
  - 2022-2025_factor_construction.parquet — sử dụng z_amihud_14d cho cost model.
  - D1_df_weightt.parquet — composite weights từ Notebook 3

Phương pháp: 
  - Portfolio Construction: Chia universe thành 3 nhóm theo composite score; long top tertile, short bottom tertile, equal-weight trong mỗi leg
  - Transaction Costs: Slippage phân tầng theo thanh khoản (z_amihud_14d) với 4 buckets: 2 / 5 / 10 / 20 bps, cộng fixed cost theo turnover.
  - Performance: Tính gross/net returns và các metrics: Annualized Return, Volatility, Sharpe, Sortino, Max Drawdown, Win Rate, Profit Factor và Newey-West t-stat.

Kết quả: 
  - Backtest OOS 2023–2025 chỉ sử dụng equal-weight, nhất quán với quyết định từ Notebook 3.
  - Mean daily return có ý nghĩa thống kê ở 5% level theo Newey-West.


