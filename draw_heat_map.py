import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)
# HEATMAP
# a = np.loadtxt('fixed_mask_after_qk.txt')
# b = np.reshape(a, (128, 320, 320))


# for i in range(1):
#     fig, ax = plt.subplots()
#     im = ax.imshow(b[i, :320, :320], cmap = 'RdBu')
#     # ax.set_xticks(np.arange(320))
#     # ax.set_yticks(np.arange(320))
#     cbar = ax.figure.colorbar(im, ax=ax)
#     fig.tight_layout()
#     # plt.show()
#     plt.savefig('qk_after_mask_image/bb/matrix-{}.eps'.format(i+1), dpi=600)
#     plt.clf()


# a = np.load('/home/songshuhui/Desktop/Transformer-0523/data/q/q-00')
# b = np.reshape(a, (128, 320, 320))

# for i in range(1):

#     c, d = np.where(b[i, :320, :320]>0)
#     plt.figure(figsize = (15,15))
#     plt.scatter(d, c, s=0.5, c='dodgerblue')
#     plt.xlim(0,320)
#     plt.ylim(320,0)

#     plt.xticks([])
#     plt.yticks([])
#     plt.show()
    # plt.savefig('qk_after_mask_image/bb_scatter/matrix-{}.eps'.format(i+1), dpi=600)
    # plt.clf()


# HEATMAP
# a = np.loadtxt('bb_mask_softmax.txt')
# b = np.reshape(a, (128, 320, 320))


# for i in range(1):
#     fig, ax = plt.subplots()
#     im = ax.imshow(b[i, :320, :320], cmap = 'Reds')
#     cbar = ax.figure.colorbar(im, ax=ax)
#     # plt.show()
#     plt.savefig('softmax_image/bb/matrix-{}.eps'.format(i+1), dpi=600)
#     plt.clf()


# SCATTER
# a = np.loadtxt('bb_mask_softmax.txt')
# b = np.reshape(a, (128, 320, 320))

# for i in range(1):

#     c, d = np.where(b[i, :320, :320]>0)
#     plt.figure(figsize = (15,15))
#     plt.scatter(d, c, s=0.5, c='tomato')
#     plt.xlim(0,320)
#     plt.ylim(320,0)

#     plt.xticks([])
#     plt.yticks([])
#     # plt.show()
#     plt.savefig('softmax_image/bb_scatter/matrix-{}.eps'.format(i+1), dpi=600)
#     plt.clf()

plt.figure(figsize=(15,15))


a = np.loadtxt('/home/songshuhui/Desktop/Transformer-0523/data/softmax_grad/softmax_grad-00.txt')
b = np.reshape(a, (40960, 320))
c, d = np.where(b[:40960, :320] != 0)

ax1 = plt.subplot(1,3,1)
plt.scatter(d, c, s=0.5, c='dodgerblue')
ax1.set_xlim(0,320)
ax1.set_ylim(40960,0)
ax1.set_xlabel('Input: softmax_grad (40960*320)', fontsize = 18)



a = np.loadtxt('/home/songshuhui/Desktop/Transformer-0523/data/k/k-00.txt')
b = np.reshape(a, (10240, 512))
c, d = np.where(b[:10240, :512] != 0)

ax2 = plt.subplot(1,3,2)
plt.scatter(d, c, s=0.5, c='dodgerblue')
ax2.set_xlim(0,512)
ax2.set_ylim(10240,0)
ax2.set_xlabel('Input: k (10240*512)', fontsize = 18)


a = np.loadtxt('/home/songshuhui/Desktop/Transformer-0523/data/q_grad/q_grad-00.txt')
b = np.reshape(a, (10240, 512))
c, d = np.where(b[:10240, :512] != 0)


ax3 = plt.subplot(1,3,3)
plt.scatter(d, c, s=0.5, c='dodgerblue')
ax3.set_xlim(0,512)
ax3.set_ylim(10240,0)
ax3.set_xlabel('Output: q_grad (10240*512)', fontsize = 18)

# plt.show()
plt.savefig('/home/songshuhui/Desktop/Transformer-0523/figure/25.q_grad.png')
