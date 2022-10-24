class Node():
    def __init__(self, pair : tuple, parent = None):
        self.person_id = pair[1]
        self.movie_id = pair[0]
        self.parent = parent
    
    def __eq__(self, other) -> bool:
        return self.person_id == other.person_id

class StackFrontier():
    def __init__(self):
        self.frontier = []

    def add(self, node):
        self.frontier.append(node)

    def contains(self, neighbor):
        return any(node == neighbor for node in self.frontier)

    def empty(self):
        return len(self.frontier) == 0

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[-1]
            self.frontier = self.frontier[:-1]
            return node


class QueueFrontier(StackFrontier):

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[0]
            self.frontier = self.frontier[1:]
            return node