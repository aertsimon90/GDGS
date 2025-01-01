import random, time, json
from collections import Counter
class GDGS_Neuron: # Generic Deep Generative System - Neuron
	def __init__(self, age=1):
		self.weight = random.random()*9
		self.bias = random.random()*9
		self.training = random.random()*9
		self.age = age
	def process(self, x, truevalue=4.5, train=True, maxing=9):
		self.age += 1
		processed_x = (x*self.weight)+self.bias
		try:
			end_x = maxing/(1+((maxing/(maxing-1))**-processed_x))
		except:
			end_x = maxing/(processed_x/(maxing/2))
		if train:
			error = truevalue-end_x
			error = error*self.training
			self.weight += error*x
			self.bias += error
			testagain = self.process(x, truevalue=truevalue, train=False)
			errornow = truevalue-testagain
			if abs(error) > abs(errornow):
				self.training -= self.training*0.01
			elif abs(error-errornow) > self.training*20:
				self.training -= self.training*0.01
			else:
				self.training += self.training*0.01
		return end_x
	def die(self, brain):
		q = self.age/brain.king.age
		if q < brain.neuron_die_age_unuse_threshold:
			return True
		else:
			return False
	def save(self):
		return {"weight": self.weight, "bias": self.bias, "training": self.training, "age": self.age}
	def load(self, data):
		self.weight = data["weight"]
		self.bias = data["bias"]
		self.training = data["training"]
		self.age = data["age"]
def value_of_text(text):
	try:
		n = []
		nn = Counter(text)
		for h in text:
			n.append((nn[h]+(ord(h)/1114112))/1.9)
		return 1/(sum(n)/len(n))
	except:
		return 0
class GDGS_Brain: # Generic Deep Generative System - Brain
	def __init__(self, neuron_count=10, clusters=9):
		self.neurons = []
		for _ in range(neuron_count):
			self.neurons.append(GDGS_Neuron())
		self.king = GDGS_Neuron()
		self.clusters = clusters
		self.new_neuron_threshold = 0.3
		self.neuron_die_age_unuse_threshold = 0.0005
	def save(self):
		neurons = []
		for h in self.neurons:
			neurons.append(h.save())
		king = self.king.save()
		return {"neurons": neurons, "king": king, "clusters": self.clusters, "new_neuron_threshold": self.new_neuron_threshold, "neuron_die_age_unuse_threshold": self.neuron_die_age_unuse_threshold}
	def load(self, data):
		neurons = []
		for h in data["neurons"]:
			n = GDGS_Neuron()
			n.load(h)
			neurons.append(n)
		king = GDGS_Neuron()
		king.load(data["king"])
		self.neurons = neurons
		self.king = king
		self.clusters = data["clusters"]
		self.new_neuron_threshold = data["new_neuron_threshold"]
		self.neuron_die_age_unuse_threshold = data["neuron_die_age_unuse_threshold"]
	def cluster(self, x, maxing=9):
		x = abs(x)
		maxing = abs(maxing)
		k = 1
		kk = maxing/abs(self.clusters)
		while True:
			if kk*k >= x:
				return k
			k += 1
			if kk*k >= maxing:
				return maxing
	def process(self, x, truevalue=4.5, train=True, maxing=9):
		results = []
		errors = []
		targetcluster = self.cluster(x, maxing=maxing)
		for neuron in self.neurons:
			q = True
			if train:
				if neuron.die(self):
					self.neurons.remove(neuron)
					q = False
			if q:
				if self.cluster(neuron.weight, maxing=maxing) == targetcluster:
					result = neuron.process(x, truevalue=truevalue, train=train, maxing=maxing)
					errors.append(abs(result-truevalue))
					results.append(result)
		if len(results) == 0:
			results = [0]
			errors = [abs(truevalue)]
		result1 = sum(results)/len(results)
		result2 = sum(results)
		resultmain = (result1+result2)/2
		kingresult = self.king.process(resultmain, truevalue=truevalue, train=train, maxing=maxing)
		errors.append(abs(kingresult-truevalue))
		errormain = (sum(errors)/len(errors))/9
		if train:
			if errormain > self.new_neuron_threshold:
				nnn = GDGS_Neuron()
				nnn.age = self.king.age
				self.neurons.append(nnn)
		return kingresult
class GDGS_Chatbot: # Generic Deep Generative System - Chatbot
	def __init__(self, neuron_count=10, clusters=9):
		self.brain = GDGS_Brain(neuron_count=neuron_count, clusters=clusters)
		self.words = {}
		self.channels = {}
		self.channel_delete_timeout = 60*10
		self.talking_model = 2 # 0=This model first uses the input data to generate a response and then uses the data of the last word used for subsequent words (Self-Continuation Context Model). 1=This model first uses the input data to generate answers, then combines the data of the last word used with the input data and uses the average for the next words (Medium Context Model). 2=This model first uses the input data to generate the entire response (Absolute Context Model) 3=A single context model using average data of all past messages and words ever used (Whole Context Model)
	def save(self):
		return {"brain": self.brain.save(), "words": self.words, "channels": self.channels, "channel_delete_timeout": self.channel_delete_timeout}
	def load(self, data):
		self.brain.load(data["brain"])
		self.words = data["words"]
		self.channels = data["channels"]
		self.channel_delete_timeout = data["channel_delete_timeout"]
	def tokenizer(self, text):
		return text.split()
	def channels_check(self):
		for ch, data in self.channels.items():
			if time.time()-data["last"] >= self.channel_delete_timeout:
				del self.channels[ch]
	def channels_still(self, ch):
		self.channels[ch]["last"] = time.time()
	def complation(self, text, truevalue=4.5, train=True, maxing=9, temperature=0.4, maxwords=15, context=[], channel=None, channelsave=True):
		wholevalues = []
		for h in context:
			h = value_of_text(h)
			h = self.brain.process(h, truevalue=truevalue, train=False, maxing=maxing)
			wholevalues.append(h)
		if channel != None:
			if channel not in self.channels:
				self.channels[channel] = {"messages": [], "last": time.time()}
			self.channels[channel]["last"] = time.time()
			context = self.channels[channel]["messages"]+context
			if channelsave:
				self.channels[channel]["messages"].append(text)
		addtext = []
		for h in self.tokenizer(text):
			for hh in context:
				if h in hh:
					if hh not in addtext:
						addtext.append(hh)
		text = " ".join(addtext)+" "+text
		if train:
			words = self.tokenizer(text)
			for word in words:
				value = value_of_text(word)
				value = self.brain.process(value, truevalue=truevalue, train=train, maxing=maxing)
				self.words[word] = value
		text_value = value_of_text(text)
		text_value = self.brain.process(text_value, truevalue=truevalue, train=train, maxing=maxing)
		main_text_value = text_value
		wholevalues.append(main_text_value)
		try:
			max_wording = 1+(int((text_value*1000000)**2)%(maxwords-1))
		except:
			max_wording = 1
		result = []
		nseed = (text_value*827462)**2
		while len(result) < max_wording:
			most = ""
			mostv = float("inf")
			targets = []
			for word, v in self.words.items():
				v = abs(v-text_value)
				if v < 1/temperature:
					targets.append(word)
				if v < mostv:
					mostv = v
					most = word
			if len(targets) == 0:
				targets.append(most)
			targetword = targets[int(nseed)%len(targets)]
			if self.talking_model == 0:
				text_value = value_of_text(targetword)
				text_value = self.brain.process(text_value, truevalue=truevalue, train=train, maxing=maxing)
			elif self.talking_model == 1:
				text_value = value_of_text(targetword)
				text_value = self.brain.process(text_value, truevalue=truevalue, train=train, maxing=maxing)
				text_value = (main_text_value+text_value+text_value)/3
			elif self.talking_model == 2:
				text_value = main_text_value
			elif self.talking_model == 3:
				h = value_of_text(targetword)
				h = self.brain.process(h, truevalue=truevalue, train=False, maxing=maxing)
				wholevalues.append(h)
				text_value = sum(wholevalues)/len(wholevalues)
			result.append(targetword)
			nseed += (nseed%30)+1
		if channelsave:
			if channel != None:
				self.channels[channel]["messages"].append(" ".join(result))
		return " ".join(result)
	def train_with_chat(self, chat, maxing=9, temperature=0.4, maxwords=15, context=[], channel=None, channelsave=True):
		messages = []
		talkings = []
		for n, message in enumerate(chat):
			try:
				next_message = chat[n+1]
			except:
				next_message = message
			value_mn = value_of_text(next_message)
			talkings.append(self.complation(message, truevalue=value_mn*9, maxing=maxing, temperature=temperature, maxwords=maxwords, train=True, context=context, channel=channel, channelsave=channelsave))
		return " ".join(talkings)
