from statjax.regression import TestRegression



test_regression = TestRegression(n_steps_ahead=1)
train_result = test_regression.train()
print("Train Result:", train_result)
reasoning_result = test_regression.reasoning()
print("Reasoning Result:", reasoning_result)
info = test_regression._info()
print("Info:", info)


