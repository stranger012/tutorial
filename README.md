# Data leakage in ML pipelines: tutorial code

Code for the data leakage series: one synthetic loan dataset, one folder per episode.

The dataset is made up. A lender's customers apply for loans between 2021 and 2024, and the task is
to predict whether a loan defaults within 12 months of its application date. The lender exported its
tables on 2026-01-01, and that export carries the same leaks real exports do. Finding and fixing them
is the point of the series.

## Quick start

```bash
uv sync    # numpy, pandas, scikit-learn; the CSVs are already in data/
```

## Tables (`data/`)

| File | Rows | One row is |
| --- | --- | --- |
| `loans.csv` | 12,448 | A loan application and its outcome. Label: `defaulted_12m` |
| `customers.csv` | 6,000 | A customer as of the export date |
| `credit_scores.csv` | 150,000 | A customer's credit score at the start of each quarter, 2020 to 2026 |
| `payments.csv` | 244,530 | One monthly instalment of a loan |
| `officer_notes.csv` | 16,269 | A note a loan officer wrote, with its date |
| `manifest.json` | | Seed, row counts and SHA-256 of each file |

The prediction moment is `loans.application_date`. Every column that is not known on that date is a leak.

### `loans.csv`

| Column | Meaning |
| --- | --- |
| `loan_id` | Loan identifier |
| `customer_id` | Customer identifier (joins to the other tables) |
| `application_date` | Date of the application: the prediction moment |
| `amount`, `term_months`, `interest_rate` | Loan terms |
| `purpose` | Why the customer is borrowing |
| `region_code`, `branch_id` | Where the loan was taken |
| `annual_income`, `employment_years`, `age` | Applicant details |
| `debt_to_income` | Existing debt payments over income |
| `region_default_rate` | Default rate of the region |
| `months_paid_first_year` | Instalments paid on time in the first 12 months |
| `last_payment_date` | Last instalment paid before the export |
| `total_recovered` | Amount recovered by collections |
| `collection_calls` | Calls made by the collections team |
| `hardship_flag` | Hardship programme (Y/N) |
| `loan_status` | Status on the export date |
| `default_date` | Date the loan defaulted, if it did |
| `defaulted_12m` | Label: 1 if the loan defaulted within 12 months |

## Find the leaks

The tables hide more than a dozen leaks. Each episode uncovers some of them; try to find them before you watch.

## Episodes

- [`01-leakage-intro/`](01-leakage-intro/): your 99% accurate model is lying (target leakage)

Coming next: `02-splits/`, `03-pipelines/`,
`04-point-in-time/`, `05-ci-gate/`, `06-llm-leakage/`.
