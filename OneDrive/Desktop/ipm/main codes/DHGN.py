import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from collections import Counter

# -----------------------------
# تعریف Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        # محاسبه cross entropy به صورت غیرتجمعی
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# -----------------------------
# تعریف Composite Loss برای ترکیب CrossEntropy و Focal Loss
class CompositeLoss(nn.Module):
    def __init__(self, weight, alpha=0.5, gamma=1.5, reduction='mean'):
        """
        alpha: وزن‌دهی بین CrossEntropy و Focal Loss.
        مثال: alpha=0.5 یعنی هر دو loss با وزن یکسان در نظر گرفته شوند.
        """
        super(CompositeLoss, self).__init__()
        self.alpha = alpha
        self.ce_loss_func = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
        self.focal_loss_func = FocalLoss(gamma=gamma, weight=weight, reduction=reduction)
    
    def forward(self, input, target):
        ce = self.ce_loss_func(input, target)
        focal = self.focal_loss_func(input, target)
        return self.alpha * ce + (1 - self.alpha) * focal

# -----------------------------
# تعریف مدل‌های پایه با کاهش سطح dropout به 0.15

class TemporalEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.15):
        super(TemporalEmbedding, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.dropout(h_n[-1])

class DynamicHypergraphConstruction(nn.Module):
    def __init__(self, input_dim, num_hyperedges):
        super(DynamicHypergraphConstruction, self).__init__()
        self.gat = nn.Linear(input_dim * 2, input_dim)
        self.fc = nn.Linear(input_dim, num_hyperedges)

    def forward(self, z):
        num_stocks = z.size(0)
        z_i = z.unsqueeze(1).repeat(1, num_stocks, 1)
        z_j = z.unsqueeze(0).repeat(num_stocks, 1, 1)
        concat = torch.cat([z_i, z_j], dim=-1)
        alpha = F.leaky_relu(self.gat(concat), negative_slope=0.2)
        alpha = torch.sum(alpha, dim=-1)
        alpha = F.softmax(alpha, dim=-1)
        m = torch.matmul(alpha, z)
        h_d = F.softmax(torch.tanh(self.fc(m)), dim=-1)
        return h_d

class HypergraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.15):
        super(HypergraphConvolution, self).__init__()
        self.theta = nn.Parameter(torch.randn(input_dim, output_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, h):
        x_tilde = F.relu(torch.matmul(x, self.theta))
        x_tilde = self.dropout(x_tilde)
        p = torch.matmul(h.T, x_tilde)
        x_prime = torch.sigmoid(torch.matmul(h, p))
        return x_prime

class IndustryRelationsAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.15):
        super(IndustryRelationsAggregator, self).__init__()
        self.fc = nn.Linear(input_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, v, h):
        v_expanded = v.unsqueeze(-1)
        h_expanded = h.unsqueeze(1)
        weighted_v = torch.sum(h_expanded * v_expanded, dim=0)
        v_agg = torch.mean(weighted_v, dim=-1)
        v_agg = v_agg.unsqueeze(0).repeat(v.size(0), 1)
        v_concat = torch.cat([v, v_agg], dim=-1)
        c = torch.tanh(self.fc(v_concat))
        return self.dropout(c)

class MultiRelationFusion(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiRelationFusion, self).__init__()
        self.num_heads = num_heads
        self.beta = nn.Parameter(torch.randn(num_heads, 2))

    def forward(self, s1, s2):
        beta_star = F.softmax(self.beta, dim=-1)
        s = torch.zeros_like(s1)
        for k in range(self.num_heads):
            s += beta_star[k, 0] * s1 + beta_star[k, 1] * s2
        s = s / self.num_heads
        return s

class DHSTN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hyperedges, num_heads, dropout=0.15):
        super(DHSTN, self).__init__()
        self.temporal = TemporalEmbedding(input_dim, hidden_dim, dropout)
        self.dhc = DynamicHypergraphConstruction(hidden_dim, num_hyperedges)
        self.hconv_dynamic = HypergraphConvolution(hidden_dim, hidden_dim, dropout)
        self.hconv_static = HypergraphConvolution(hidden_dim, hidden_dim, dropout)
        self.ira = IndustryRelationsAggregator(hidden_dim, hidden_dim, dropout)
        self.mrf = MultiRelationFusion(hidden_dim, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, h_static):
        z = self.temporal(x)
        h_dynamic = self.dhc(z)
        c_dynamic = self.ira(z, h_dynamic)
        s1 = self.hconv_dynamic(z, h_dynamic) * c_dynamic
        c_static = self.ira(z, h_static)
        s2 = self.hconv_static(z, h_static) * c_static
        s = self.mrf(s1, s2)
        combined = torch.cat([z, self.dropout(s)], dim=-1)
        y_hat = torch.softmax(self.output(combined), dim=-1)
        return y_hat

# -----------------------------
# بارگذاری و پیش‌پردازش داده‌های واقعی
def load_real_data():
    csi300_ticker = "000300.SS"  # نمایه CSI300
    nasdaq100_ticker = "^NDX"    # نمایه NASDAQ100

    csi300_data = yf.download(csi300_ticker, start="2019-12-01", end="2022-12-30")
    nasdaq100_data = yf.download(nasdaq100_ticker, start="2020-03-01", end="2023-03-30")

    def compute_features(df):
        features = pd.DataFrame()
        features["Open"] = df["Open"]
        features["Close"] = df["Close"]
        features["High"] = df["High"]
        features["Low"] = df["Low"]
        features["Volume"] = df["Volume"]

        features["MA5"] = features.Close.rolling(window=5).mean()
        features["MA10"] = features.Close.rolling(window=10).mean()
        features["MA20"] = features.Close.rolling(window=15).mean()
        features["MA30"] = features.Close.rolling(window=30).mean()

        features["RESI5"] = features.Close.pct_change().rolling(window=5).sum()
        features["RSQR5"] = (features.Close.pct_change() ** 2).rolling(window=5).sum()
        features["KLEN"] = features.High - features.Low
        features["KLOW"] = features.Open - features.Low
        features["RET20"] = features.Close.pct_change(periods=20)
        features["ROC60"] = features.Close.pct_change(periods=60)
        features["VSTD5"] = features.Volume.rolling(window=5).std()
        features["BIAS20"] = (features.Close - features.MA20) / features.MA20
        features["STD5"] = features.Close.pct_change().rolling(window=5).std()
        features["WVMA5"] = (features.Close * features.Volume).rolling(window=5).mean() / features.Volume.rolling(window=5).mean()

        features = features.dropna()
        return features, df["Close"]

    csi300_features, csi300_close = compute_features(csi300_data)
    nasdaq100_features, nasdaq100_close = compute_features(nasdaq100_data)

    def to_tensor(df, seq_len=30):
        data = torch.tensor(df.values, dtype=torch.float32)
        num_samples = len(data) - seq_len + 1
        x = torch.zeros((num_samples, seq_len, data.shape[1]))
        for i in range(num_samples):
            x[i] = data[i:i+seq_len]
        return x

    x_csi300 = to_tensor(csi300_features)
    x_nasdaq100 = to_tensor(nasdaq100_features)

    y_csi300 = (csi300_features.Close.shift(-1) > csi300_features.Close).astype(int).iloc[29:-1]
    y_nasdaq100 = (nasdaq100_features.Close.shift(-1) > nasdaq100_features.Close).astype(int).iloc[29:-1]

    dates_csi300 = csi300_features.index[29:-1]
    dates_nasdaq100 = nasdaq100_features.index[29:-1]

    # همگام‌سازی قیمت‌ها با داده‌های دیگر
    csi300_close = csi300_close[29:-1]
    nasdaq100_close = nasdaq100_close[29:-1]

    num_hyperedges = 10
    h_static_csi300 = torch.randint(0, 2, (x_csi300.shape[0], num_hyperedges)).float()
    h_static_nasdaq100 = torch.randint(0, 2, (x_nasdaq100.shape[0], num_hyperedges)).float()

    return (x_csi300, h_static_csi300, torch.tensor(y_csi300.values, dtype=torch.long), dates_csi300, csi300_close), \
           (x_nasdaq100, h_static_nasdaq100, torch.tensor(y_nasdaq100.values, dtype=torch.long), dates_nasdaq100, nasdaq100_close)

# -----------------------------
# آموزش و ارزیابی
def train_and_evaluate(model, x, h_static, y_true, num_epochs=100):
    class_counts = Counter(y_true.numpy())
    print(f"Class distribution: {class_counts}")
    total = sum(class_counts.values())
    weights = torch.tensor([total / (2 * class_counts[i]) for i in range(2)], dtype=torch.float32)
    
    # استفاده از Composite Loss به جای تنها focal یا cross entropy
    loss_func = CompositeLoss(weight=weights, alpha=0.5, gamma=1.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(x, h_static)
        loss = loss_func(y_pred, y_true)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        y_prob = model(x, h_static)
        y_pred = torch.argmax(y_prob, dim=1).numpy()
        y_prob = y_prob.numpy()
        y_true_np = y_true.numpy()
    return evaluate_metrics(y_true_np, y_pred, y_prob), y_true_np, y_pred

def evaluate_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob[:, 1])
    return {"ACC": acc, "PRE": pre, "REC": rec, "F1": f1, "AUC": auc}

# -----------------------------
# رسم نمودارها
def plot_metrics_comparison(metrics_dict, title):
    methods = list(metrics_dict.keys())
    aucs = [metrics_dict[m]["AUC"] * 100 for m in methods]
    accs = [metrics_dict[m]["ACC"] * 100 for m in methods]
    f1s = [metrics_dict[m]["F1"] * 100 for m in methods]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].bar(methods, aucs)
    ax[0].set_title("AUC")
    ax[0].tick_params(axis='x', rotation=45)
    ax[1].bar(methods, accs)
    ax[1].set_title("ACC")
    ax[1].tick_params(axis='x', rotation=45)
    ax[2].bar(methods, f1s)
    ax[2].set_title("F1")
    ax[2].tick_params(axis='x', rotation=45)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_backtesting_performance(y_true, y_pred, title):
    returns = np.where(y_pred == y_true, 0.01, -0.01)
    cumulative_returns = np.cumsum(returns)
    plt.plot(cumulative_returns, label="DHSTN")
    plt.title(f"Backtesting Performance - {title}")
    plt.xlabel("Days")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.show()

def plot_actual_vs_predicted(close_prices, y_true, y_pred, dates, title):
    # تبدیل close_prices به آرایه یک‌بعدی
    close_prices = close_prices.values.flatten() if isinstance(close_prices, pd.Series) else close_prices.to_numpy().flatten()
    
    # چاپ ابعاد برای دیباگ
    print(f"Shape of close_prices: {close_prices.shape}")
    print(f"Shape of y_true: {y_true.shape}")
    print(f"Shape of y_pred: {y_pred.shape}")
    print(f"Shape of dates: {len(dates)}")

    # محاسبه بازده روزانه واقعی
    actual_returns = pd.Series(close_prices).pct_change().fillna(0).values
    print(f"Shape of actual_returns: {actual_returns.shape}")
    actual_cumulative_returns = np.cumsum(actual_returns)
    print(f"Shape of actual_cumulative_returns: {actual_cumulative_returns.shape}")

    # محاسبه بازده پیش‌بینی‌شده
    predicted_returns = np.where(y_pred == y_true, actual_returns, -actual_returns)
    print(f"Shape of predicted_returns: {predicted_returns.shape}")
    predicted_cumulative_returns = np.cumsum(predicted_returns)
    print(f"Shape of predicted_cumulative_returns: {predicted_cumulative_returns.shape}")

    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual_cumulative_returns, label="Actual Cumulative Returns", color="blue", linestyle="-")
    plt.plot(dates, predicted_cumulative_returns, label="Predicted Cumulative Returns (DHSTN)", color="red", linestyle="--")
    plt.title(f"Cumulative Returns - {title}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# -----------------------------
# اجرای کد اصلی
input_dim = 19
hidden_dim = 512
output_dim = 2
num_hyperedges = 10
num_heads = 6

# بارگذاری داده‌ها
(x_csi300, h_static_csi300, y_true_csi300, dates_csi300, csi300_close), \
(x_nasdaq100, h_static_nasdaq100, y_true_nasdaq100, dates_nasdaq100, nasdaq100_close) = load_real_data()

# مدل‌ها
model_csi300 = DHSTN(input_dim, hidden_dim, output_dim, num_hyperedges, num_heads, dropout=0.15)
model_nasdaq100 = DHSTN(input_dim, hidden_dim, output_dim, num_hyperedges, num_heads, dropout=0.15)

# آموزش و ارزیابی (در اینجا آخرین 200 نمونه برای هر مجموعه استفاده می‌شود)
metrics_csi300, y_true_csi300_eval, y_pred_csi300 = train_and_evaluate(
    model_csi300, x_csi300[-200:], h_static_csi300[-200:], y_true_csi300[-200:], num_epochs=100
)
metrics_nasdaq100, y_true_nasdaq100_eval, y_pred_nasdaq100 = train_and_evaluate(
    model_nasdaq100, x_nasdaq100[-200:], h_static_nasdaq100[-200:], y_true_nasdaq100[-200:], num_epochs=100
)

# نتایج مقاله برای مقایسه
metrics_dict_csi300 = {
    "LSTM": {"AUC": 0.5086, "ACC": 0.5064, "F1": 0.5238},
    "Transformer": {"AUC": 0.5091, "ACC": 0.5061, "F1": 0.5060},
    "DHSTN": metrics_csi300
}

metrics_dict_nasdaq100 = {
    "LSTM": {"AUC": 0.5007, "ACC": 0.5010, "F1": 0.5429},
    "Transformer": {"AUC": 0.5016, "ACC": 0.5022, "F1": 0.5294},
    "DHSTN": metrics_nasdaq100
}

# چاپ نتایج
print("CSI300 Metrics:", metrics_csi300)
print("NASDAQ100 Metrics:", metrics_nasdaq100)

# رسم نمودارها
plot_metrics_comparison(metrics_dict_csi300, "CSI300")
plot_metrics_comparison(metrics_dict_nasdaq100, "NASDAQ100")
plot_backtesting_performance(y_true_csi300_eval, y_pred_csi300, "CSI300")
plot_backtesting_performance(y_true_nasdaq100_eval, y_pred_nasdaq100, "NASDAQ100")
plot_actual_vs_predicted(csi300_close[-200:], y_true_csi300_eval, y_pred_csi300, dates_csi300[-200:], "CSI300")
plot_actual_vs_predicted(nasdaq100_close[-200:], y_true_nasdaq100_eval, y_pred_nasdaq100, dates_nasdaq100[-200:], "NASDAQ100")