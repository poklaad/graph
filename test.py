FILENAME = 'CA-AstroPh.txt'
LANDMARKS_COUNT = range(20, 101, 40)
import datetime as tm
import random
import csv
import collections

#-------------------запись графа-----------------------------
from operator import truediv

def read_from_txt(filename):
    graph = {}
    with open(filename, encoding = 'ansi') as file:

        row = file.readline()
        while row:

            parent, child = row.split()
            parent = int(parent)
            child = int(child)

            if parent in graph:
                if child not in graph[parent]['linked']:
                    graph[parent]['linked'].append(child)
                    graph[parent]['degree'] += 1
            else:
                graph[parent] = {
                    'linked': [child],
                    'degree': 1,
                    'visit': '',
                    'shortest_paths':{},
                }


            if child in graph:
                if parent not in graph[child]['linked']:
                    graph[child]['linked'].append(parent)
                    graph[child]['degree'] += 1

            else:
                graph[child] = {
                    'linked': [parent],
                    'degree': 1,
                    'visit': '',
                    'shortest_paths':{},
                }

            row = file.readline()

    return graph


#-------------------------bfs--------------------------------
def bfs_color(start, finish, graph, marks):
    tic = tm.datetime.now()
    for i in graph:
        graph[i]['visit'] = 0
    for i in marks:
        graph[i]['visit'] = 1
    graph[start]['visit'] = 1
    found = False
    queue = [start]
    pathes = {}
    pathes [start] = [start]
    while queue:
        a = queue[0]
        queue.pop(0)
        for i in graph[a]['linked']:
            if graph[i]['visit'] == 0:
                graph[i]['visit'] = 1
                queue.append(i)
                p = pathes[a].copy()
                p.append(i)
                pathes[i] = []
                pathes[i] = p
                if i == finish:
                    found = True
                    path = pathes[i]
                    return path, (tm.datetime.now() - tic).total_seconds()

    return -1, (tm.datetime.now() - tic).total_seconds()

def bfs_color1(start, finish, graph, marks):
    for i in graph:
        graph[i]['visit'] = 0
    for i in marks:
        graph[i]['visit'] = 1
    graph[start]['visit'] = 1
    found = False
    queue = [start]
    pathes = {}
    pathes [start] = [start]
    while queue:
        a = queue[0]
        queue.pop(0)
        for i in graph[a]['linked']:
            if graph[i]['visit'] == 0:
                graph[i]['visit'] = 1
                queue.append(i)
                p = pathes[a].copy()
                p.append(i)
                pathes[i] = []
                pathes[i] = p
                if i == finish:
                    found = True
                    path = pathes[i]
                    graph[start]['shortest_paths'][i] = path
                    return path

    graph[start]['shortest_paths'][finish] = -1
    return -1

#------------------------------------------------------------


def landmarks_basic(start, finish, marks, graph):
    tic = tm.datetime.now()
    size_min_path = len(graph) + 1
    min_path = []
    for i in marks:
        path1 = graph[start]['shortest_paths'][i]
        
        path2 = graph[finish]['shortest_paths'][i]

        if path1 != -1 and path2 != -1:
            if len(path1) + len(path2) < size_min_path:
                min_path = path1 + path2[1:]
                size_min_path = len(min_path)

    if min_path == []:
        return -1, (tm.datetime.now() - tic).total_seconds()
    else:
        return min_path, (tm.datetime.now() - tic).total_seconds()




#------------------------------------------------------------


def landmarks_bfs(start, finish, graph, marks):
    start_time= tm.datetime.now()
    little_graph = {}
    start_path = graph[start]['shortest_paths']
    finish_path = graph[finish]['shortest_paths']

    for mark, path in list(start_path.items()) + list(finish_path.items()):
        if path == -1 or path == []:
            continue
        if path[len(path)-1] not in little_graph:
            little_graph[path[len(path)-1]] = { 
                'linked': [],
                'visit': ''
            }
        for v in range(len(path)-1):
            if path[v] in little_graph and path[v+1] not in little_graph[path[v]]['linked']:
                little_graph[path[v]]['linked'].append(path[v+1])
            elif path[v] not in little_graph:
                little_graph[path[v]] = {
                    'linked': [path[v+1]], 
                    'visit': ''
                }


            if path[v+1] in little_graph and path[v] not in little_graph[path[v+1]]['linked']:
                little_graph[path[v+1]]['linked'].append(path[v])
            elif path[v+1] not in little_graph:
                little_graph[path[v+1]] = {
                    'linked': [path[v]],
                    'visit': ''
                }
    if len(little_graph) == 0 or start not in little_graph or finish not in little_graph:
        return -1, (tm.datetime.now() - start_time).total_seconds()
    else:
        path, _ = bfs_color(start, finish, little_graph, [])
        return path, (tm.datetime.now() - start_time).total_seconds()



def landmarks_pathes (start, finish, graph, marks):
    tic = tm.datetime.now()
    for i in marks:
        path1 = bfs_color1(start, i, graph, [])
        path2 = bfs_color1(finish, i, graph, [])
        if path2 != -1:
            path2.reverse()
    return (tm.datetime.now() - tic).total_seconds()


#--------------------выбор марок----------------------------
def choose_random(nodes, count_landmarks):
    start_time= tm.datetime.now()
    marks = random.sample(nodes, count_landmarks)
    return marks, (tm.datetime.now() - start_time).total_seconds()

def choose_dergee(nodes, count_landmarks):
    start_time= tm.datetime.now()
    marks = nodes[:count_landmarks]
    return marks, (tm.datetime.now() - start_time).total_seconds()

def choose_coverege(nodes, count_landmarks, graph):
    start_time= tm.datetime.now()
    number_of_uses = {}
    while len(number_of_uses) < landmarks_count:
        uses_part = {}
        rand_nod = random.sample(nodes, 2*count_landmarks)
        start_nod = rand_nod[:landmarks_count]
        finish_nod = rand_nod[landmarks_count:]
        for i, j in zip(start_nod, finish_nod):
            #used_nodes - список с номерами вершин, которые попали в кратчайший путь
            used_nodes, _ = bfs_color(i,j, graph,[])
            if used_nodes == -1:
                continue
            for v in used_nodes:
                if v not in uses_part:
                    uses_part[v] = 1
                else:
                    uses_part[v] += 1
        number_of_uses = {**number_of_uses, **uses_part}
    number_of_uses = sorted(number_of_uses, reverse=True)
    number_of_uses = number_of_uses[:landmarks_count]
    return number_of_uses, (tm.datetime.now() - start_time).total_seconds()

#------------------------------------------------------------



#------------------------------------------------------------


def landmarks_choose (graph, nodes, selection, count_landmarks):
    if selection == 'random':
        marks, timer_landmarks = choose_random(nodes, count_landmarks)
    elif selection == 'degree':
        marks, timer_landmarks = choose_dergee(nodes, count_landmarks)
    else:
        marks, timer_landmarks = choose_coverege(nodes, count_landmarks, graph)
    return marks, timer_landmarks

#------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#основная часть
import random
import networkx as nx
import collections

marks_selection = ('random', 'degree', 'coverege')
estimation_strategies = ('basic', 'landmarks_bfs')

"""[здесь будет словарь с ключом - количество марок и значениями: точность и время работы]"""
results = {'basic':{
                    'random': {}, 
                    'degree': {},
                    'coverege': {}
                 },
        'landmarks_bfs': {
            'random': {}, 
            'degree': {},
            'coverege': {}
            },
        'bfs': 0,
        'choose_landmarks': {
            'random': {}, 
            'degree': {},
            'coverege': {}
            }
         }
for i in LANDMARKS_COUNT:
    for j in marks_selection:
        results['basic'][j][i] = {'time': 0, 'accuracy': 0}
        results['landmarks_bfs'][j][i] = {'time': 0, 'accuracy': 0}
        results['choose_landmarks'][j][i] = {'time': 0}


with open('results_1.txt', 'w') as file:
    

    graph = read_from_txt(FILENAME)
    graph_items = graph.items()
    tic = tm.datetime.now()
    sort_degree = sorted(graph_items, key=lambda x: x[1]['degree'], reverse=True) 
    nodes = [i[0] for i in sort_degree]
    tic = tm.datetime.now() - tic
    graph_size = len(nodes)
    



    number_of_tests = 1


    #выбор начальных и конечных вершин

    nodes_random = nodes.copy()
    random.shuffle(nodes_random)
    nodes_start = nodes_random[:number_of_tests]
    nodes_finish = nodes_random[graph_size - number_of_tests:]



    #для тестов с заданными вершинами

    #test(graph, 1, nodes, 20, 'random',  [9481, 69904, 16093, 122877, 32410, 104288, 92189, 55511, 105439, 33781, 80576, 58060, 70830, 18009, 105595, 89838, 30224, 84205, 95646, 124599], 24842 , 102498)
    #test(graph, 1, nodes, 20, 'random')



    #запуск алгоритмов с разными параметрами
    for start, finish in zip(nodes_start,nodes_finish):

        if start == finish:
            if start < graph_size-1:
                start += 1
            else:
                start -= 1

        print ('start and finish: ', start, ' ', finish)

        s_path, timer_exact = bfs_color(start, finish, graph, [])
        results['bfs'] += timer_exact
        
        print ('bfs done, time: ', timer_exact)

        for selection in marks_selection:
            for landmarks_count in LANDMARKS_COUNT:


                landmarks, timer_landmarks = landmarks_choose (graph, nodes, selection, landmarks_count)
                landmarks_pathes (start, finish, graph, landmarks)
                results['choose_landmarks'][selection][landmarks_count]['time'] += timer_landmarks
                if selection == 'degree':
                    results['choose_landmarks'][selection][landmarks_count]['time'] += tic.total_seconds()

                print ('landmarks done, time: ', timer_landmarks)

                path_bfs, timer_bfs = landmarks_bfs (start, finish, graph, landmarks)
                results['landmarks_bfs'][selection][landmarks_count]['time'] += timer_bfs

                print ('landmarks_bfs done, time: ', timer_bfs)

                path_basic, timer_basic = landmarks_basic (start, finish, landmarks, graph)
                results['basic'][selection][landmarks_count]['time'] += timer_basic

                print ('landmarks_basic done, time: ', timer_basic)

                if path_bfs != -1:
                    approximation_error_bfs = (len(path_bfs) - len(s_path))/len(s_path)
                    results['landmarks_bfs'][selection][landmarks_count]['accuracy'] += approximation_error_bfs

                if path_basic != -1:
                    approximation_error_basic = (len(path_basic) - len(s_path))/len(s_path)
                    results['basic'][selection][landmarks_count]['accuracy'] += approximation_error_basic

for selection in marks_selection:
    for landmarks_count in LANDMARKS_COUNT:
        results['basic'][selection][landmarks_count]['accuracy'] /= number_of_tests
        results['landmarks_bfs'][selection][landmarks_count]['accuracy'] /= number_of_tests
        results['choose_landmarks'][selection][landmarks_count]['time'] /= number_of_tests
        results['landmarks_bfs'][selection][landmarks_count]['time'] /= number_of_tests
        results['basic'][selection][landmarks_count]['time'] /= number_of_tests
results['bfs'] /= number_of_tests
            
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt




x = []
y_basic_random_time = []
y_basic_degree_time = []
y_basic_coverege_time = []
y_landmarks_bfs_random_time = []
y_landmarks_bfs_degree_time = []
y_landmarks_bfs_coverege_time = []
y_bfs_random_time = []
y_bfs_degree_time = []
y_bfs_coverege_time = []
y_choose_landmarks_random_time = []
y_choose_landmarks_degree_time = []
y_choose_landmarks_coverege_time = []

y_basic_random_accuracy = []
y_basic_degree_accuracy = []
y_basic_coverege_accuracy = []
y_landmarks_bfs_random_accuracy = []
y_landmarks_bfs_degree_accuracy = []
y_landmarks_bfs_coverege_accuracy = []
y_bfs_random_accuracy = []
y_bfs_degree_accuracy = []
y_bfs_coverege_accuracy = []
y_choose_landmarks_random_accuracy = []
y_choose_landmarks_degree_accuracy = []
y_choose_landmarks_coverege_accuracy = []

for count in LANDMARKS_COUNT:
    x.append(count)
    y_basic_random_time.append(results['basic']['random'][count]['time'])
    y_basic_degree_time.append(results['basic']['degree'][count]['time'])
    y_basic_coverege_time.append(results['basic']['coverege'][count]['time'])###########################################################
    y_landmarks_bfs_random_time.append(results['landmarks_bfs']['random'][count]['time'])
    y_landmarks_bfs_degree_time.append(results['landmarks_bfs']['degree'][count]['time'])
    y_landmarks_bfs_coverege_time.append(results['landmarks_bfs']['coverege'][count]['time'])###########################################################
    y_bfs_random_time.append(results['bfs'])
    y_bfs_degree_time.append(results['bfs'])
    y_bfs_coverege_time.append(results['bfs'])###########################################################
    y_choose_landmarks_random_time.append(results['choose_landmarks']['random'][count]['time'])
    y_choose_landmarks_degree_time.append(results['choose_landmarks']['degree'][count]['time'])
    y_choose_landmarks_coverege_time.append(results['choose_landmarks']['coverege'][count]['time'])###########################################################

    y_basic_random_accuracy.append(results['basic']['random'][count]['accuracy'])
    y_basic_degree_accuracy.append(results['basic']['degree'][count]['accuracy'])
    y_basic_coverege_accuracy.append(results['basic']['coverege'][count]['accuracy'])###########################################################
    y_landmarks_bfs_random_accuracy.append(results['landmarks_bfs']['random'][count]['accuracy'])
    y_landmarks_bfs_degree_accuracy.append(results['landmarks_bfs']['degree'][count]['accuracy'])
    y_landmarks_bfs_coverege_accuracy.append(results['landmarks_bfs']['coverege'][count]['accuracy'])###########################################################



fig, ax = plt.subplots()
plt.plot(x, y_basic_random_time, color='blue')
plt.plot(x, y_basic_degree_time, color='red')
plt.plot(x, y_basic_coverege_time, color='green')
plt.xlabel('Количество марок')
plt.ylabel('Время работы')
plt.title('Время работы basic алгоритма на марках для разных стратегий', fontsize=30)
plt.legend(['random landmarks', 'degree landmarks', 'coverage landmarks'])


fig, ax = plt.subplots()
plt.plot(x, y_landmarks_bfs_random_time, color='blue')
plt.plot(x, y_landmarks_bfs_degree_time, color='red')
plt.plot(x, y_landmarks_bfs_coverege_time, color='green')
plt.xlabel('Количество марок')
plt.ylabel('Время работы')
plt.title('Время работы landmarks_bfs алгоритма на марках для разных стратегий', fontsize=30)
plt.legend(['random landmarks', 'degree landmarks', 'coverage landmarks'])


fig, ax = plt.subplots()
plt.plot(x, y_bfs_random_time, color='blue')
plt.plot(x, y_bfs_degree_time, color='red')
plt.plot(x, y_bfs_coverege_time, color='green')
plt.xlabel('Количество марок')
plt.ylabel('Время работы')
plt.title('Время работы простого bfs алгоритма на марках для разных стратегий', fontsize=30)
plt.legend(['random landmarks', 'degree landmarks', 'coverage landmarks'])


fig, ax = plt.subplots()
plt.plot(x, y_choose_landmarks_random_time, color='blue')
plt.plot(x, y_choose_landmarks_degree_time, color='red')
plt.plot(x, y_choose_landmarks_coverege_time, color='green')
plt.xlabel('Количество марок')
plt.ylabel('Время работы')
plt.title('Время выбора марок для разных стратегий', fontsize=30)
plt.legend(['random landmarks', 'degree landmarks', 'coverage landmarks'])


fig, ax = plt.subplots()
plt.plot(x, y_basic_random_accuracy, color='blue')
plt.plot(x, y_basic_degree_accuracy, color='red')
plt.plot(x, y_basic_coverege_accuracy, color='green')
plt.xlabel('Количество марок')
plt.ylabel('Точность')
plt.title('Точность basic алгоритма на марках для разных стратегий', fontsize=30)
plt.legend(['random landmarks', 'degree landmarks', 'coverage landmarks'])


fig, ax = plt.subplots()
plt.plot(x, y_landmarks_bfs_random_accuracy, color='blue')
plt.plot(x, y_landmarks_bfs_degree_accuracy, color='red')
plt.plot(x, y_landmarks_bfs_coverege_accuracy, color='green')
plt.xlabel('Количество марок')
plt.ylabel('Точность')
plt.title('Точность landmarks_bfs алгоритма на марках для разных стратегий', fontsize=30)
plt.legend(['random landmarks', 'degree landmarks', 'coverage landmarks'])




plt.show()
