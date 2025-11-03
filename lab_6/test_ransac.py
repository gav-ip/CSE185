import numpy as np
import numpy.linalg as la
from skimage.measure import ransac

class LeastSquareLine:
    def __init__(self):
        self.a = 0.0
        self.b = 0.0

    def estimate(self, points2D):
        B = points2D[:,1]
        A = np.copy(points2D)
        A[:,1] = 1.0
        
        # Solve least squares: A * [a, b]^T = B
        result = np.linalg.inv(A.T @ A) @ A.T @ B
        self.a = result[0]
        self.b = result[1]
        return True

    def predict(self, x): 
        return (self.a * x) + self.b

    def predict_y(self, x): 
        return (self.a * x) + self.b

    def residuals(self, points2D):
        return points2D[:,1] - self.predict(points2D[:,0])

    def line_par(self):
        return self.a, self.b

# Generate data like in the notebook
np.random.seed(seed=1)

# True line parameters
a_true, b_true = 0.2, 20.0

# Generate noisy line data
x_start, x_end = -200.0, 200.0
x = np.arange(x_start,x_end)
y = a_true * x + b_true
data = np.column_stack([x, y])

# Add noise
noise = np.random.normal(size=data.shape)
data += 5 * noise
data[::2] += 10 * noise[::2]
data[::4] += 20 * noise[::4]

# Add outliers
faulty = np.array(30 * [(180., -100)])
faulty += 5 * np.random.normal(size=faulty.shape)
data[:faulty.shape[0]] = faulty

print(f"True line: a={a_true}, b={b_true}")
print(f"Data shape: {data.shape}")
print(f"Outliers: 30, Inliers: {data.shape[0]-30}")
print(f"Inlier ratio w: {(data.shape[0]-30)/data.shape[0]:.3f}")

# Test 1: Least squares (should fail due to outliers)
print("\n=== Test 1: Least Squares (no RANSAC) ===")
LSline = LeastSquareLine()
LSline.estimate(data)
a_ls, b_ls = LSline.line_par()
print(f"Least Squares result: a={a_ls:.4f}, b={b_ls:.4f}")
print(f"Error: Δa={abs(a_ls-a_true):.4f}, Δb={abs(b_ls-b_true):.4f}")

# Test 2: RANSAC with different thresholds
print("\n=== Test 2: RANSAC with different thresholds ===")
for threshold in [50, 70, 90, 110]:
    model_robust, inliers = ransac(data, LeastSquareLine, 
                                    min_samples=2, 
                                    residual_threshold=threshold, 
                                    max_trials=20)
    a_rs, b_rs = model_robust.line_par()
    num_inliers = np.sum(inliers)
    print(f"Threshold={threshold}: a={a_rs:.4f}, b={b_rs:.4f}, inliers={num_inliers}/{data.shape[0]}")
    print(f"  Error: Δa={abs(a_rs-a_true):.4f}, Δb={abs(b_rs-b_true):.4f}")

# Test 3: Check residuals for true line
print("\n=== Test 3: Residual analysis for true line ===")
true_residuals = []
for i in range(30, data.shape[0]):  # Skip outliers (first 30)
    residual = abs(data[i,1] - (a_true * data[i,0] + b_true))
    true_residuals.append(residual)

print(f"Inlier residuals (from true line):")
print(f"  Mean: {np.mean(true_residuals):.2f}")
print(f"  Median: {np.median(true_residuals):.2f}")
print(f"  Std: {np.std(true_residuals):.2f}")
print(f"  95th percentile: {np.percentile(true_residuals, 95):.2f}")
print(f"  99th percentile: {np.percentile(true_residuals, 99):.2f}")
print(f"  Max: {np.max(true_residuals):.2f}")








