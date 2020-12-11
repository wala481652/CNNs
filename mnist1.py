import random


from mnist import MNIST

mndata = MNIST('CNNs/bin')
mndata.gz = True

images, labels = mndata.load_training()

index = random.randrange(0, len(images)) 
print(mndata.display(images[index]))