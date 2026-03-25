# R&D Test

The aim of this exercise is to assess your reasoning skills.

To perform this test, please fork the repository and open a pull request when you're done. (Please tell us by email when you're ready for review! :smiley:).

## Objective
Given an undirected graph, you need to compute a path which travels each edge at least one time (pseudo-eulerian path), total distance traveled (sum of edges' weight) being the smallest possible.

⚠️ the resulting path must cover every **edge** !!

## Input
A few graph instances are available in a folder named `instances`.

### Format
- First line of file contains 2 numbers: first is total number of vertices, second is total number of edges in the graph.
- Then vertices are listed: the 2 numbers given are the vertex coordinates in a 2-D plan. Vertex id is 0 for the first given vertex, then, 1, 2... and so on.
- Then are listed the edges, with 3 numbers being: id of 1st vertex, id of 2nd vertex, weight associated with their edge.

## Assessment criteria
- You must provide an answer for all given instances. Answer must be given in reasonable time (on a standard performance computer).
- Current code is provided to get started easily. You may want to edit it, or not using it. Feel free to arrange the project as you wish.
- Please include a file `SOLUTION.md` presenting roughly your approach.

## Project setup
- This project uses `python3`.
- Install requirements with running `pip install -r requirements.txt`.
- You can then run `python main.py -h` to see usage.

## Advices
- We have provided a very basic and incomplete example solution
    - Feel free to correct/improve it or start a new solution from scratch. Initiatives are valued !
- Do not spend more than 3 hours on this test.
- Do not hesitate to submit a partial rendering, flagging with `# TODO` items you didn't have time to complete.


Good luck!


# Comment on the solution

The solution is based on the construction of a semi-eulerian graph, by doubling
the edges starting from odd-degree vertices with the smallest weights.

In the current version, it is still non-optimal for the following reasons:
- the minimum weight perfect matching relies on a greedy solution
- large graphs take a long time to be processed because of the repetition of
dijkstra executions
- separated graphs (as in the islands example) are not handled.