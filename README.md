# Travelling salesman problem solving

The Travelling Salesman Problem (TSP) is a well-known problem in the field of operational research. It involves finding the shortest possible route to visit a given set of cities, starting and ending at the same city, and visiting each city only once. The problem can be represented as a graph, with cities represented as vertices and the distances between cities represented as weights on the edges.

TSP is a NP-hard problem. 

## Setup 

To setup the project, you need a virtual environment using conda or virtualenv, with at least python 3 :
```
virtualenv venv
```

Then you need to source it :
```
source venv/bin/activate
```

Then install all the requirements :
```
pip install -r requirements
```

## Run

To run the project you need to tun this command with these options :
```
python tsp_solver.py -n N_NODE --s WIDTH_GRID_SIZE HEIGHT_GRID_SIZE -v VERBOSE -t TIMER
```
`N_NODE` is the number of node, default is 30

`WIDTH_GRID_SIZE` is the width of the grid size, default is 150

`HEIGHT_GRID_SIZE` is the height of the grid size, default is 150

`VERBOSE` is a boolean for having an output after each iteration, default is False

`TIMER` is a boolean for having the execution time, default is False. If verbose is active, then timer won't be active

## Examples

This is the default use, with a grid size of 150x150 and 30 nodes
```
python tsp_solver.py
```

This is second example, with a grid size of 100x100 and 30 nodes
```
python tsp_solver.py -s 100 100
```

This is a third example, with a grid size of 150x150 and 40 nodes
```
python tsp_solver.py -n 40
```

## Warning

This program is working on my computer `MacBook Air M2 8-core CPU 10-core GPU 24gb RAM` but might not finish in reasonable time on yours.
