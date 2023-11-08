import base64
from typing import Union
import streamlit as st
import pandas as pd

from quickbooks import QuickBooks
from quickbooks.objects.customer import Customer
from intuitlib.enums import Scopes
from intuitlib.client import AuthClient
from quickbooks.objects.salesreceipt import SalesReceipt
from quickbooks.objects.invoice import Invoice
from quickbooks.objects.deposit import Deposit
from quickbooks.objects.company_info import CompanyInfo
import logging
import time
import seaborn as sns
import matplotlib.pyplot as plt
from prophet import Prophet


CLIENT_ID = st.secrets["app_conf"]["CLIENT_ID"]
CLIENT_SECRET = st.secrets["app_conf"]["CLIENT_SECRET"]

QB_TOKEN_SECRETS_DATA = st.secrets.get("qb_token")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)


def get_refresh_token_expiration():

    last_refresh_time = QB_TOKEN_SECRETS_DATA.get("last_time_refresh_token")
    if last_refresh_time is None:
        return None

    refresh_exp = int(last_refresh_time) + 8640000  # 100 days in seconds

    logging.info("refresh token expiration: " + str(refresh_exp))

    return refresh_exp


def get_access_token_expiration() -> Union[int, None]:

    last_acc_token_time = QB_TOKEN_SECRETS_DATA.get("last_time_access_token")
    if last_acc_token_time is None:
        return None
    access_exp = int(last_acc_token_time) + 3600

    logging.info("access token expiration: " + str(access_exp))

    return access_exp


def get_client():
    """
    get_client: used for refreshing tokens and creating a new client

    returns:
        QuickBooks client
    """
    auth_client = AuthClient(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        # If you do not pass this in, the Quickbooks client will
        # call refresh and get a new access token.
        environment="sandbox",
        redirect_uri="https://developer.intuit.com/v2/OAuth2Playground/RedirectUrl",
    )

    if QB_TOKEN_SECRETS_DATA is None:
        st.error("No token data found. Please contact caleb@depotanalytics.co to resolve this issue.")

    refresh_token = QB_TOKEN_SECRETS_DATA.get("refresh_token")
    company_id = QB_TOKEN_SECRETS_DATA.get("company_id")
    access_token_exp = get_access_token_expiration()
    refresh_token_exp = get_refresh_token_expiration()

    curr_time = time.time()
    if curr_time < access_token_exp:
        # access token still valid, refresh also valid
        print("Access token is valid")
    elif curr_time < refresh_token_exp and refresh_token is not None:
        # access token invalid, refresh valid
        print("Refresh token is valid")
        auth_client.refresh(refresh_token=refresh_token)
    else:
        # access & refresh are invalid
        st.error("No tokens are currently valid. Please contact caleb@depotanalytics.co to resolve this issue.")

    client = QuickBooks(
        auth_client=auth_client,
        refresh_token=refresh_token,
        company_id=company_id,
    )

    return client

# Authenticate and connect to QuickBooks Online account
if "qbo" not in st.session_state:
    qbo = get_client()
    st.session_state.qbo = qbo
qbo = st.session_state.qbo

# Query QuickBooks Online API to retrieve data
if "customers" not in st.session_state:
    customers = Customer.all(qb=qbo)
    st.session_state.customers = pd.DataFrame.from_records(
        [c.to_dict() for c in customers]
    )
df_customers = st.session_state.customers

if "sales" not in st.session_state:
    sales = SalesReceipt.all(qb=qbo)
    st.session_state.sales = pd.DataFrame.from_records([s.to_dict() for s in sales])
df_sales = st.session_state.sales
df_sales["Date"] = pd.to_datetime(df_sales["TxnDate"])
df_sales.set_index("Date", inplace=True)

if "invoices" not in st.session_state:
    invoices = Invoice.all(qb=qbo)
    st.session_state.invoices = pd.DataFrame.from_records(
        [i.to_dict() for i in invoices]
    )
df_invoices = st.session_state.invoices
df_invoices["Date"] = pd.to_datetime(df_invoices["TxnDate"])
df_invoices.set_index("Date", inplace=True)

if "deposits" not in st.session_state:
    deposits = Deposit.all(qb=qbo)
    st.session_state.deposits = pd.DataFrame.from_records(
        [d.to_dict() for d in deposits]
    )
df_deposits = st.session_state.deposits
df_deposits["Date"] = pd.to_datetime(df_deposits["TxnDate"])
df_deposits.set_index("Date", inplace=True)

# Project revenues for the next 2 years
if "sales_revenues" not in st.session_state:
    sales_revenues = (
        df_sales["TotalAmt"].resample("M").sum()
    )  # Monthly revenues from sales
    st.session_state.sales_revenues = sales_revenues
sales_revenues = st.session_state.sales_revenues

if "invoices_revenues" not in st.session_state:
    invoices_revenues = (
        df_invoices["TotalAmt"].resample("M").sum()
    )  # Monthly revenues from invoices
    st.session_state.invoices_revenues = invoices_revenues
invoices_revenues = st.session_state.invoices_revenues

if "deposits_revenues" not in st.session_state:
    deposits_revenues = (
        df_deposits["TotalAmt"].resample("M").sum()
    )  # Monthly revenues from deposits
    st.session_state.deposits_revenues = deposits_revenues
deposits_revenues = st.session_state.deposits_revenues

# combine sales and invoices TotalAmt and Date columns
if "df_revenues" not in st.session_state:
    df_revenues = pd.concat([df_invoices, df_sales, df_deposits])
    st.session_state.df_revenues = df_revenues
df_revenues = st.session_state.df_revenues

if "revenues" not in st.session_state:
    revenues = (
        pd.concat([sales_revenues, invoices_revenues, deposits_revenues])
        .groupby(level=0)
        .sum()
    )  # Combine revenues from sales and invoices
    st.session_state.revenues = revenues
revenues = pd.DataFrame(data=st.session_state.revenues.reset_index()).rename(
        columns={"Date": "ds", "TotalAmt": "y"}
    )

# Get QB company file name
if "company_name" not in st.session_state:
    company_name = CompanyInfo.get(id=1, qb=qbo)
    st.session_state.company_name = company_name


"""Dashboard GUI Begins Here"""

# Create Streamlit app to display data and projections
st.title("Meta Prophet Demo Dashboard")
st.write("This dashboard uses the Meta Prophet package to project revenues for the next 2 years. Note that parameters are critical and must be tuned correctly to get meaningful projections.")
st.write("Find more information about Meta Prophet for forecasting here: https://facebook.github.io/prophet/")

demo_type = st.selectbox("Select Demo Type", ("Upload Data", "Quickbooks Sample Data"))
if demo_type == "Upload Data":
    # write out the requirements for the CSV file
    st.write("Upload a CSV file with columns 'ds' and 'y' where 'ds' is a date and 'y' is a numeric value representing the total dollar value of a transaction. The dates can be in any order/months can be missing/etc...")
    st.session_state.uploaded_file = st.file_uploader("Upload CSV", type="csv")

if demo_type == "Upload Data" and ("uploaded_file" not in st.session_state):
    st.stop()
elif demo_type == "Upload Data" and ("uploaded_file" in st.session_state):
    if st.session_state.uploaded_file is None:
        st.stop()


if demo_type == "Quickbooks Sample Data":
    st.write(f"### {st.session_state.company_name}: QuickBooks Revenue Projection")
    with st.expander("Customer Dataframe", expanded=False):
        st.write(df_customers)

    with st.expander("Previous Revenues (Monthly Sums)", expanded=False):
        st.write(revenues)
elif demo_type == "Upload Data":
    st.write(f"### {st.session_state.uploaded_file.name}: Uploaded Data Revenue Projection")
    revenues = pd.read_csv(st.session_state.uploaded_file)
    try:
        revenues['ds'] = pd.to_datetime(revenues['ds'])
    except:
        st.error("Please upload a CSV file with a column named 'ds' that contains dates.")
        st.stop()
    try:
        revenues['y'] = pd.to_numeric(revenues['y'])
    except:
        st.error("Please upload a CSV file with a column named 'y' that contains numeric values.")
        st.stop()


"""Revenue Projection - Creating GUI for Prophet Parameters"""
seasonality_mode = st.selectbox(
    "Select seasonality mode",
    ("additive", "multiplicative"),
)

weekly_seasonality = st.checkbox("Weekly seasonality", value=True)
daily_seasonality = st.checkbox("Daily seasonality", value=False)
confidence_interval = st.slider("Confidence Interval for the Predicted Range", 0.0, 1.0, 0.8)
seasonality_prior_scale = st.number_input("Seasonality prior scale", value=14)
growth_type = st.selectbox(
    "Select growth type",
    ("linear", "logistic"),
    index=0,
)

if growth_type == "logistic":
    cap = st.number_input("Monthly Saturation Capacity", value=int(max(revenues['y'])*3))
    floor = st.number_input("Monthly Saturation Floor", value=0)
else:
    cap = None
    floor = None

revenues['cap'] = [cap]*len(revenues)
revenues['floor'] = [floor]*len(revenues)

# Create Prophet model and fit to revenues data
model = Prophet(
    seasonality_mode=seasonality_mode,
    weekly_seasonality=weekly_seasonality,
    daily_seasonality=daily_seasonality,
    interval_width=confidence_interval,
    growth=growth_type,
    holidays_prior_scale=seasonality_prior_scale,
    seasonality_prior_scale=seasonality_prior_scale,
)

model.fit(revenues)

# Project revenues for the next 2 years
future = model.make_future_dataframe(periods=24, freq="M")
future['cap'] = [cap]*len(future)
future['floor'] = [floor]*len(future)
revenues_proj = model.predict(future).tail(24)

# turn the index into a date column
revenues_proj.index = pd.date_range(start=revenues.iloc[-1]['ds'], periods=24, freq="M")

# Visualize data and projections using Seaborn
fig, ax = plt.subplots()
sns.set_style("darkgrid")
sns.lineplot(data=revenues.reset_index(), x="ds", y="y", label="Actual", ax=ax)
sns.lineplot(
    data=revenues_proj.reset_index(),
    x="index",
    y="yhat",
    label="Projected",
    ax=ax,
)
sns.lineplot(
    data=revenues_proj.reset_index(),
    x="index",
    y="yhat_upper",
    label="Projected Upper Bound",
    ax=ax,
)
sns.lineplot(
    data=revenues_proj.reset_index(),
    x="index",
    y="yhat_lower",
    label="Projected Lower Bound",
    ax=ax,
)
ax.fill_between(
    revenues_proj.index,
    revenues_proj["yhat_lower"],
    revenues_proj["yhat_upper"],
    alpha=0.3,
)
ax.set_xlabel("Date")
# make xlabels diagonal so they don't overlap
ax.tick_params(axis="x", labelrotation=45)
ax.set_ylabel("Revenue")
ax.set_title("Revenue Projection for Next 2 Years")
ax.legend()

st.pyplot(fig)


data_display_cols = st.columns(2)

with data_display_cols[0]:
    st.write("#### Table: Historical Revenue Data")
    st.write(revenues)

with data_display_cols[1]:
    st.write("#### Table: Revenue Projection for Next 2 Years")
    st.write(revenues_proj)

st.header("Want to use prophet for forecasting in your organization? We'd love to help - schedule time to talk with me!")
st.write("https://calendly.com/caleb-da/30-minute-meeting-daily")
