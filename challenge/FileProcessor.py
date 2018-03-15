# Author: WeldFire
# Created: 12/20/2016
"""
This function loads a pokemon csv file and parses out labels 
and stat information for input into a machine learning algorithm

IN:
filepath - the filepath of the csv you are wanting to load and parse

OUT:
pokemonStatLines - The number values of one pokemons stats
pokemonTypeLabels - The digitized pokemon type1 labels
trackedTypes - The key value pair transition table used to convert types to IDs
"""
def loadCleanData(filepath):
	#Read all lines of the dataset into memory
	with open(filepath) as file:
		lines = file.readlines()
	
	#Grab the titles from the first line of the dataset 
	titles = lines[1]
	dataLines = lines[1:]
	
	#Placeholder for each type that we would like to track
	#(This will help us ID our training labels)
	trackedTypes = {}
	typeID = 0
	
	#Define our output variables
	pokemonStatLines = []
	pokemonTypeLabels = []
	
	#Loop through each data line to parse and clean it
	for line in dataLines:
		pokemonStats = line.split(',')
		
		#Column 2 holds the type 1 data
		pokemonType1 = pokemonStats[2]
		
		if pokemonType1 not in trackedTypes:
			#ID our type if it isn't already
			trackedTypes[pokemonType1] = typeID
			#Increase our ID for the next iteration
			typeID = typeID + 1
		
		#Placeholder for one Pokemons stats
		pokemonStatLine = []
		
		#Loop through each of the stats that we deem to be the most important in determining type
		#For now lets only grab HP, Attack, Defense, Sp. Atk, Sp. Def, Speed, and Generation
		for i in range(5, 12):
			pokemonStatLine.append(pokemonStats[i]) 
		
		#Add our generated data to our output variables
		pokemonStatLines.append(pokemonStatLine)
		pokemonTypeLabels.append(trackedTypes[pokemonType1])
			
	return pokemonStatLines, pokemonTypeLabels, trackedTypes

"""
Provided a tracked types key value pair (KVP) it will return the type text

IN:
ID - ID you are wanting to find the type name for
trackedTypes - The KVP type map to search inside of

OUT:
type - The plain text representation of the pokemons type
"""
def typeFromTrackedTypeID(ID, trackedTypes):
	for type, typeID in trackedTypes.iteritems():
		if ID == typeID:
			return type
	return "UNKNOWN TYPE"