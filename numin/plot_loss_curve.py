import matplotlib.pyplot as plt

train_loss = [
    0.2372, 0.2274, 0.2233, 0.2203, 0.2176, 0.2148, 0.2116, 0.2097, 0.2070, 0.2046,
    0.2021, 0.1996, 0.1973, 0.1915, 0.1889, 0.1871, 0.1859, 0.1841, 0.1827, 0.1781,
    0.1766, 0.1752, 0.1742, 0.1725, 0.1726, 0.1702, 0.1689, 0.1685, 0.1678, 0.1671,
    0.1672, 0.1649, 0.1652, 0.1647, 0.1641, 0.1645, 0.1636
]

val_loss = [
    0.3903, 0.3822, 0.3808, 0.3773, 0.3770, 0.3726, 0.3690, 0.3721, 0.3698, 0.3718,
    0.3719, 0.3741, 0.3720, 0.3750, 0.3789, 0.3785, 0.3783, 0.3819, 0.3834, 0.3873,
    0.3845, 0.3882, 0.3925, 0.3934, 0.3936, 0.3958, 0.3935, 0.3926, 0.3946, 0.3911,
    0.3984, 0.3998, 0.3992, 0.3996, 0.4041, 0.3990, 0.4001
]

epochs = list(range(1, len(train_loss) + 1))

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Train Loss', marker='o', markersize=4)
plt.plot(epochs, val_loss, label='Validation Loss', marker='s', markersize=4)

plt.axvline(x=7, color='r', linestyle='--', label='Best Validation Loss (Epoch 7)', alpha=0.6)

plt.title('Stock Prediction Model: Learning Curve (Loss)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

plt.savefig('loss_curve.png', dpi=300)
print("Plot saved as loss_curve.png")
