import tensorflow as tf
import numpy as np
from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import sklearn as sk
from collections import Counter

filename = "qifeng_data_npfea_sep09_smote_30mfea.csv"

X = []
Y = []
data = open(filename, "r")
count = 0
line_count = 0
neg_examples = 0
pos_examples = 0
for line in data:
	if line_count == 0:
		line_count += 1
		continue

	l = line.strip().split(',')
	x = []
	
	for i in l[:-1]:
		x.append(float(i))
	y = float(l[-1])

	if y == 0.0:
		neg_examples += 1
	else:
		pos_examples += 1


	if pos_examples == 5000 and neg_examples == 5000:
		break

	if pos_examples < 5000 and y == 1.0:
		X.append(x)
		Y.append(y)

	if neg_examples < 5000 and y == 0.0:
		X.append(x)
		Y.append(y)




		



	#if count > 1000:
	#	break
	count += 1



X = np.array(X)
Y = np.array(Y)
Y = np.array([Y, -(Y-1)]).T
X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.30)

# Parameters
learning_rate = 0.0001
training_epochs = 500
batch_size = 1000
display_step = 1

# Network Parameters
n_hidden_1 = 70 # 1st layer number of features
n_hidden_2 = 70 # 2nd layer number of features
#n_hidden_3 = 50 # 2nd layer number of features
#n_hidden_4 = 20 # 2nd layer number of features
n_input = len(X[0]) # Number of feature
n_classes = 2 # Number of classes to predict

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Hidden layer with RELU activation
    #layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    #layer_3 = tf.nn.relu(layer_3)
    # Hidden layer with RELU activation
    #layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    #layer_4 = tf.nn.relu(layer_4)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    #out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return out_layer


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    #'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    #'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes])),
    #'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    #'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    #'b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
#cross_entropy = tf.reduce_sum(- y * tf.log(pred) - (1 - y) * tf.log(1 - pred), 1)
#cost = tf.reduce_mean(-tf.reduce_sum(pred * tf.log(y), reduction_indices=[1]))
#cross_entropy = tf.reduce_sum(pred*tf.log(y + 1e-10))
#cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(X)/batch_size)
        X_batches = np.array_split(X, total_batch)
        Y_batches = np.array_split(Y, total_batch)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = X_batches[i], Y_batches[i]
            # batch_y.shape = (batch_y.shape[0], 1)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost)
    print "Optimization Finished!"

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "Accuracy:", accuracy.eval({x: X_test, y: Y_test})
    global result 
    result = tf.argmax(pred, 1).eval({x: X_test, y: Y_test})

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run(accuracy, feed_dict={x: X_test, y: Y_test}))

    y_p = tf.argmax(pred, 1)

    #prediction_probab = pred/tf.reduce_sum(pred,1)


    #print ("may be probabilities "), prediction_probab.eval(session=sess,feed_dict={x:pred}) #prediction_probab.eval()


    val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x:X_test, y:Y_test})

    Y_test = [not i for i in Y_test[:,0]]    
    y_pred =  y_pred.tolist()

    (test_precision,test_recall,test_fscore,test_support)=precision_recall_fscore_support(Y_test, y_pred, beta=1.0, labels=None,
	                            pos_label=1, average=None,
	                            warn_for=('precision', 'recall',
	                                      'f-score'),
	                            sample_weight=None)

    Test = (test_precision,test_recall,test_fscore,test_support)
    print "TEST RESULT:"
    print"=============="
    print "\n"
    print "PRECISION:"
    print(Test[0])
    print "\n"
    print "RECALL:"
    print(Test[1])
    print "\n"
    print "F1-score:"
    print(Test[2])
    print "\n"	
    print "SUPPORT:"
    print(Test[3])



