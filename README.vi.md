**Đọc bằng ngôn ngữ khác:** [English](README.md) | [Tiếng Việt](README.vi.md)

# CROSS-SECTIONAL FACTOR INVESTING FOR CRYPTO MARKET

Đây là một báo cáo nghiên cứu việc đầu tư dựa vào các yếu tố giữa các tài sản cho thị trường Crypto- Xây dựng một chiến lược long-short dựa trên việc xếp hạng các đồng coin theo các đặc trưng (factors) định lượng thay vì dự đoán giá tuyệt đối

Mục tiêu cuối của dự án không phải là tối đa hóa một con số Sharpe đẹp, mà là chứng minh được toàn bộ quy trình từ khâu chọn universe → tạo factor → kiểm định thống kê → xây alpha → backtest có thực hiện đúng kỷ luật của một quy trình nghiên cứu định lượng chuyên nghiệp hay không: tách bạch giai đoạn khám phá khỏi giai đoạn kiểm định ngoài mẫu (out-of-sample), tránh look-ahead bias, và định lượng được chi phí giao dịch thực tế.

Pipeline gồm 4 notebook, thực hiện tuần tự và phụ thuộc dữ liệu lẫn nhau:

[1] Universe + Raw Factors  →  [2] Statistical Validation  →  [3] Composite Alpha  →  [4] Portfolio & Backtest

## Cấu Trúc dự án

|  | Notebook                       | Giai đoạn             | Input                                                  | Output                                                                                           |
| - | ------------------------------ | --------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| 1 | `1_Universe_RawFactors.ipynb`  | Universe & Raw Factor | Dữ liệu thị trường gốc                                 | Universe & raw factors sạch, point-in-time, chia thành 2020–2021 (In Sample) và 2022–2025 (Out of Sample) |
| 2 | `2_StatisticalFiltering.ipynb` | Statistical Filtering | Raw factors — **2020–2021 (IS)**                       | Các factor có bằng chứng thống kê; không sử dụng dữ liệu 2022–2025                           |
| 3 | `3_WeightingScheme.ipynb`      | Weighting Scheme      | Các factor đã lọc + dữ liệu **2022–2025**              | Phương pháp weighting bằng walk-forward: Tối ưu trọng số trên quá khứ- áp dụng cho tương lai- dùng cửa sổ trượt về phía trước và lặp lại để kết hợp 2 factor |
| 4 | `4_Execution_Cost.ipynb`       | Execution & Cost      | Alpha + weighting scheme + dữ liệu **2022–2025 (OOS)** | Danh mục thực thi, transaction costs, turnover và hiệu suất OOS                           |

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

    



