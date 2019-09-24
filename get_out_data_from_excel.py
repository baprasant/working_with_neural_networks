###################################
file = open("neural_network_dataset/train/Y_Train.txt", "w")
for i in range(1,361):
	if i<=180:
		file.write("1\n")
	else:
		file.write("2\n")

file.close()
###################################
###################################
file = open("neural_network_dataset/test/Y_Test.txt", "w")
for i in range(1,41):
	if i<=20:
		file.write("1\n")
	else:
		file.write("2\n")

file.close()
###################################
