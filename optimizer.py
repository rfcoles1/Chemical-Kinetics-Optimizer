import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import os
import sys

Ri = 4
colourplot = 1

sys.path.insert(0,'./reactions')
reactionfile = 'eq' + str(Ri)
react = __import__(reactionfile)

outfile = open('OptimizeResults.dat',"w")

class agent():
    def __init__(self, lr, s_size, a_size):

        self.state_in = tf.placeholder(shape=[1,s_size], dtype=tf.int32)
        OHstate = slim.one_hot_encoding(self.state_in, s_size)
        #one fully connected layer
        output = slim.fully_connected(OHstate, a_size, biases_initializer = None, activation_fn = tf.nn.sigmoid, weights_initializer = tf.ones_initializer())
		
        self.output = tf.reshape(output,[-1])
        self.chosen_action = tf.argmax(self.output,0)

        self.reward_holder = tf.placeholder(shape=[1], dtype = tf.float32)
        self.action_holder = tf.placeholder(shape=[1], dtype = tf.int32)

        self.responsible_weight = tf.slice(self.output, self.action_holder, [1])

        self.loss = -(tf.log(self.responsible_weight)*self.reward_holder)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        self.update = optimizer.minimize(self.loss)

class reaction():
    def __init__(self):
        self.num_rates = react.get_Nk()		
        self.num_actions = self.num_rates*2 + 1	# +/- each constant and no action
        self.rates = np.ones(self.num_rates)
        self.inputA = 100
        self.inputB = 100
        self.inputC = 100
        self.inputE = 100		

        self.timelimit = 1
        self.maxK = 2
        self.minK = 0
        self.increment = 0.01

    def change_rate(self,action,index):
		#see reaction files 
        currProd = react.reaction(self.inputA, self.inputB, self.inputC, self.rates[0], self.rates[1], self.rates[2], self.timelimit)

        #action comes in as an int representing +/- one of the reaction rate constants 
        for i in range(0,self.num_rates):
            if (action == i) & (self.rates[i] < self.maxK):
                self.rates[i] += self.increment		
            if (action == (i+self.num_rates)) & (self.rates[i] > self.minK):
                self.rates[i] -= self.increment

        nextProd = react.reaction(self.inputA, self.inputB, self.inputC, self.rates[0], self.rates[1], self.rates[2], self.timelimit)

        #if the prod is increased, reward is +1, thus only incouraging increases
        diff = nextProd - currProd
        if diff <= 0:
            rew = -1
        else:
            rew = 1

        return [rew, nextProd]

    def getRates(self):
        return self.rates				


tf.reset_default_graph()

React = reaction()
Agent = agent(lr = 0.001, s_size = React.num_rates, a_size = React.num_actions)  
weights = tf.trainable_variables()

total_frames = 5000
total_product = 0
avg_product = 0

ei = 1.
ef = 0.1
Numfor_e = total_frames/10
h = (ef - ei)/Numfor_e
e = ei

init = tf.global_variables_initializer()

Results = np.zeros([React.num_rates+1,total_frames])
ProductGraph = np.zeros(total_frames)

with tf.Session() as sess:
    sess.run(init)
    
    for i in range (1,total_frames+1):
	
        s = React.getRates()
		
        if np.random.rand() < e:
            action = np.random.randint(React.num_actions)
        else:	
            action = sess.run(Agent.chosen_action, feed_dict={Agent.state_in:[s]})
	
        reward,product = React.change_rate(action,i)

        feed_dict = {Agent.reward_holder:[reward], Agent.action_holder:[action], Agent.state_in:[s]}
        _,ww = sess.run([Agent.update,weights], feed_dict=feed_dict)

        if e > ef:
            e += h
        
        #storing data for plotting 
        total_product += product
        avg_product += product
        if i%1000 == 0:
            print (i, s, (total_product/i), (avg_product/1000))	
            avg_product = 0
			
        Results[0,i-1] = product
        for j in range(React.num_rates):
            Results[j+1,i-1] = s[j]
		
np.savetxt(outfile,np.transpose(Results), delimiter='\t', newline='\n')
outfile.close()
