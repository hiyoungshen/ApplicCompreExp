#  提交结果
submission = pd.read_csv('sample_submission.csv')
predict = net(test_features)
predict = predict.detach().squeeze().numpy()
submission['SalePrice'] = predict
submission.to_csv('torch_submission.csv', index=False)