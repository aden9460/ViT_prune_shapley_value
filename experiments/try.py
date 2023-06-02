import numpy as np
# # iu = np.mask_indices(3, np.triu)
# # a = np.arange(9).reshape(3, 3)
# # # print(np.triu([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1))
# # b = np.mask_indices.append(1)
# # print(a)
# # print(a[iu])
# # t = [1,2,3,4,5]
# # print(t[1:3])
# from itertools import combinations, permutations
# # mask_indices = []
# # # for i in range(495):
# # #     mask_indices = list(combinations([1, 2, 3,4,5,6,7,8,9,10,11,12], 8))[i]
# # #     print(mask_indices)
# # n=5
# # np.random.seed(0)
# # for i in np.random.permutation(n):
# #     print(i)
# import models.cifar10 as cifar10
# from sklearn.model_selection import KFold
# a=0
# a += 1
# print(a)
# X = [1,2,3,4,5,6,7,8,9,10]
# Y = [11,12,13,14,15,16,17,18,19,20]
# x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
# y = np.array([0, 1, 0, 1])
# x = []
# for i in range(4):
#     x.append(i)
# print(x)
# X,V,Y= cifar10.get_dataset_and_loaders(val_split=0, val_batch_size=64)
# print('train len:{}'.format(len(X)))
# kfold=KFold(n_splits = 4,shuffle = False)
# for train_index, val_index in kfold.split(x):
#     print('train_index:{}'.format(train_index))
#     print('val_index:{}'.format(val_index))
# # accuracies = [1,2,3,4,5,6]
# way = 'good'
# mean_accuracy = np.mean(np.array(accuracies))
# print("{} Mean Accuracy: {}".format(way,mean_accuracy))

# b = [1,2]
# b=np.array(b)
# # b =+b
# # b=b.tolist()
# # b /= 3
# b=np.divide(b,3)
# # b=b.tolist()
# b = b.astype(float)
# print(b)
for j in range(4):
    for i in np.random.permutation(4):
        print(i)