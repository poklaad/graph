FILENAME = 'test_socfb-Reed98.txt'
LANDMARKS_COUNT = range (20, 101, 40)
import datetime as tm
import time
import random
import csv
import collections
import numpy as np
import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

#-------------------запись графа-----------------------------
from operator import truediv

def read_from_txt(filename):
    graph = {}
    with open(filename, encoding = 'ansi') as file:

        row = file.readline()
        while row:

            if filename != 'vk2.txt':
                parent, child = row.split()
            else:
                parent, child, _, __ = row.split()
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
                    'shortest_pathes':{},
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
                    'shortest_pathes':{},
                }

            row = file.readline()

    return graph
#------------------------------------------------------------

#-------------------------bfs--------------------------------
def bfs_color(start, finish, graph, write = False):
    tic = time.perf_counter()
    for i in graph:
        graph[i]['visit'] = 0
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
                    if write:
                        graph[start]['shortest_pathes'][i] = path
                    return path, (time.perf_counter() - tic)
    if write:
        graph[start]['shortest_pathes'][finish] = -1
    return -1, (time.perf_counter() - tic)
#------------------------------------------------------------

#-----------------------basic--------------------------------
def landmarks_basic(start, finish, marks, graph):
    tic = time.perf_counter()
    size_min_path = len(graph) + 1
    min_path = []
    for i in marks:
        path1 = graph[start]['shortest_pathes'][i]
        path2 = graph[finish]['shortest_pathes'][i]

        if path1 != -1 and path2 != -1:
            if len(path1) + len(path2) - 1 < size_min_path:
                min_path = path1 + path2[1:]
                size_min_path = len(min_path)

    if min_path == []:
        return -1, (time.perf_counter() - tic)
    else:
        return min_path, (time.perf_counter() - tic)
#------------------------------------------------------------

#--------------------landmark bfs----------------------------
def landmarks_bfs(start, finish, graph, marks):
    start_time= time.perf_counter()
    little_graph = {}
    start_path = graph[start]['shortest_pathes']
    finish_path = graph[finish]['shortest_pathes']

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
        return -1, (time.perf_counter() - start_time)
    else:
        path, _ = bfs_color(start, finish, little_graph)
        return path, (time.perf_counter() - start_time)
#------------------------------------------------------------

#---------------поиск путей (для подграфа)-------------------
def landmarks_pathes (start, finish, graph, marks):
    tic = time.perf_counter()
    for i in marks:
        path1, time_ = bfs_color(start, i, graph, True)
        path2, time_ = bfs_color(finish, i, graph, True)
        if path2 != -1:
            path2.reverse()
    return (time.perf_counter() - tic)
#-----------------------------------------------------------

#--------------------выбор марок----------------------------
def choose_random(nodes, count_landmarks):
    start_time= time.perf_counter()
    marks = random.sample(nodes, count_landmarks)
    return marks, (time.perf_counter() - start_time)

def choose_dergee(nodes, count_landmarks):
    start_time= time.perf_counter()
    marks = nodes[:count_landmarks]
    return marks, (time.perf_counter() - start_time)

def choose_coverege(nodes, landmarks_count, graph):
    start_time= time.perf_counter()
    number_of_uses = {}
    while len(number_of_uses) < landmarks_count:
        uses_part = {}
        rand_nod = random.sample(nodes, 2*landmarks_count)
        start_nod = rand_nod[:landmarks_count]
        finish_nod = rand_nod[landmarks_count:]
        for i, j in zip(start_nod, finish_nod):
            #used_nodes - список с номерами вершин, которые попали в кратчайший путь
            used_nodes, _ = bfs_color(i,j, graph)
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
    return number_of_uses, (time.perf_counter() - start_time)

def landmarks_choose (graph, nodes, selection, count_landmarks):
    if selection == 'random':
        marks, timer_landmarks = choose_random(nodes, count_landmarks)
    elif selection == 'degree':
        marks, timer_landmarks = choose_dergee(nodes, count_landmarks)
    else:
        marks, timer_landmarks = choose_coverege(nodes, count_landmarks, graph)
    return marks, timer_landmarks
#------------------------------------------------------------

#-------------------функция для тестов-----------------------
def test(file, results, graph, number_of_tests, start = '', finish = '', selections = ''):
    count_landmarks = LANDMARKS_COUNT
    marks_selection = ['random', 'degree', 'coverege']
    graph_items = graph.items()
    tic = time.perf_counter()
    sort_degree = sorted(graph_items, key=lambda x: x[1]['degree'], reverse=True) 
    nodes = [i[0] for i in sort_degree]
    tic = time.perf_counter() - tic
    graph_size = len(nodes)
    if selections != '':
        marks_selection = selections
    if start == '':
        nodes_random = nodes.copy()
        random.shuffle(nodes_random)
        nodes_start = nodes_random[:number_of_tests]
        nodes_finish = nodes_random[graph_size - number_of_tests:]
    else:
        nodes_start = [start]
        nodes_finish = [finish]
    i = 1
    for start, finish in zip(nodes_start,nodes_finish):
        #if start == finish:
        #    if start < graph_size-1:
        #        start += 1
        #    else:
        #        start -= 1
        print(i)
        print ('start and finish: ', start, ' ', finish)
        file.write(str(i))
        i += 1
        file.write('. start and finish: ')
        file.write(str(start)) 
        file.write(' ') 
        file.write(str(finish)) 
        file.write('\n')
        s_path, timer_exact = bfs_color(start, finish, graph)
        results['BFS'] += timer_exact
        
        print ('bfs done, time: ', timer_exact)
        file.write('BFS \n\t path: ') 
        file.write(str(s_path)) 
        file.write('\n\t time: ') 
        file.write(str(timer_exact)) 
        file.write('\n')
        file.write('Landmark-BFS and Landmark-basic \n')

        for selection in marks_selection:
            for landmarks_count in count_landmarks:

                file.write('\t selection: ') 
                file.write(selection) 
                file.write('\n\t count landmarks: ') 
                file.write(str(landmarks_count)) 
                file.write('\n')
                landmarks, timer_landmarks = landmarks_choose(graph, nodes, selection, landmarks_count)
                timer_pathes = landmarks_pathes(start, finish, graph, landmarks)
                results['Finding pathes'][selection][landmarks_count]['Time, sec'] += timer_pathes
                results['Landmarks selection'][selection][landmarks_count]['Time, sec'] += timer_landmarks
                if selection == 'degree':
                    results['Landmarks selection'][selection][landmarks_count]['Time, sec'] += tic

                file.write('\t landmarks: ')
                file.write(str(landmarks))
                file.write('\n\t time of choice landmarks: ')
                file.write(str(timer_landmarks))
                file.write('\n\t time of search pathes: ')
                file.write(str(timer_pathes))
                
                print ('landmarks done, time: ', timer_landmarks)
                path_bfs, timer_bfs = landmarks_bfs (start, finish, graph, landmarks)
                results['Landmarks-BFS'][selection][landmarks_count]['Time, sec'] += timer_bfs

                print ('landmarks_bfs done, time: ', timer_bfs)
                file.write('\n\t Landmark-BFS \n\t\t path: ')
                file.write(str(path_bfs)) 
                file.write('\n\t\t time: ')
                file.write(str(timer_bfs))

                path_basic, timer_basic = landmarks_basic (start, finish, landmarks, graph)
                results['Basic'][selection][landmarks_count]['Time, sec'] += timer_basic

                print ('landmarks_basic done, time: ', timer_basic)
                file.write('\n\t Landmark-basic \n\t\t path: ')
                file.write(str(path_basic))
                file.write('\n\t\t time: ')
                file.write(str(timer_basic))
                file.write('\n\n')

                if path_bfs != -1:
                    approximation_error_bfs = (len(path_bfs) - len(s_path))/len(s_path)
                    results['Landmarks-BFS'][selection][landmarks_count]['Approximation error'] += approximation_error_bfs

                if path_basic != -1:
                    approximation_error_basic = (len(path_basic) - len(s_path))/len(s_path)
                    results['Basic'][selection][landmarks_count]['Approximation error'] += approximation_error_basic

    file.write('\n\n\n Average result')
    results['BFS'] /= number_of_tests
    file.write('\n\t BFS time: ')
    file.write(str(results['BFS']))
    for selection in marks_selection:
        file.write('\n\t ')
        file.write(selection)
        for landmarks_count in count_landmarks:
            file.write('\n\t\t ')
            file.write(str(landmarks_count))
            file.write(' landmarks')
            results['Basic'][selection][landmarks_count]['Approximation error'] /= number_of_tests
            results['Landmarks-BFS'][selection][landmarks_count]['Approximation error'] /= number_of_tests
            results['Landmarks selection'][selection][landmarks_count]['Time, sec'] /= number_of_tests
            results['Landmarks-BFS'][selection][landmarks_count]['Time, sec'] /= number_of_tests
            results['Basic'][selection][landmarks_count]['Time, sec'] /= number_of_tests
            results['Finding pathes'][selection][landmarks_count]['Time, sec'] /= number_of_tests
            file.write('\n\t\t\t choose landmark time: ')
            file.write(str(results['Landmarks selection'][selection][landmarks_count]['Time, sec']))
            file.write('\n\t\t\t basic: accuracy: ')
            file.write(str(results['Basic'][selection][landmarks_count]['Approximation error']))
            file.write('\t time: ')
            file.write(str(results['Basic'][selection][landmarks_count]['Time, sec']))
            file.write('\n\t\t\t landmark-bfs: accuracy: ')
            file.write(str(results['Landmarks-BFS'][selection][landmarks_count]['Approximation error']))
            file.write('\t time: ')
            file.write(str(results['Landmarks-BFS'][selection][landmarks_count]['Time, sec']))
#------------------------------------------------------------

#-------------------построение графиков-----------------------
def create_histogram(results, method, characteristic, marks_selection):
    name = str(method) + '_' + str(characteristic) + '.png'
    step = (LANDMARKS_COUNT[-1] - LANDMARKS_COUNT[0])/max((len(LANDMARKS_COUNT)-1),1)
    x1 = np.array(LANDMARKS_COUNT) - step/4
    x2 = np.array(LANDMARKS_COUNT) + step/4
    x3 = np.array(LANDMARKS_COUNT)
    bins = np.array(LANDMARKS_COUNT)
    y1 = []
    y2 = []
    y3 = []

    for count in LANDMARKS_COUNT:
        if len(marks_selection) == 1:
            y1.append(results[method][marks_selection[0]][count][characteristic])
        elif len(marks_selection) == 2:
            y1.append(results[method][marks_selection[0]][count][characteristic])
            y2.append(results[method][marks_selection[1]][count][characteristic])
        else:
            y1.append(results[method][marks_selection[0]][count][characteristic])
            y2.append(results[method][marks_selection[1]][count][characteristic])
            y3.append(results[method][marks_selection[2]][count][characteristic])

    if step == 0:
        step = 5
    fig, ax = plt.subplots()
    if len(marks_selection) == 1:
        ax.bar(x3, y1, width = step/4, label=str(marks_selection[0]))
    elif len(marks_selection) == 2:
        ax.bar(x1 + step/8, y1, width = step/4, label=str(marks_selection[0]))
        ax.bar(x2 - step/8, y2, width = step/4, label=str(marks_selection[1]))
    else:
        ax.bar(x1, y1, width = step/4, label=str(marks_selection[0]))
        ax.bar(x3, y2, width = step/4, label=str(marks_selection[1]))
        ax.bar(x2, y3, width = step/4, label=str(marks_selection[2]))
    plt.xticks(bins)
    ax.legend(fontsize = 14)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.xlim([LANDMARKS_COUNT[0] - step, LANDMARKS_COUNT[-1] + step])
    plt.xlabel('Number of Landmarks', fontsize = 20)
    plt.ylabel(str(characteristic), fontsize = 20)
    plt.title(str(method), fontsize=30)
    fig.set_figwidth(12)    #  ширина Figure
    fig.set_figheight(7)    #  высота Figure
    fig.savefig(name)

def create_plot(results, method, characteristic, marks_selection):
    name = str(method) + '_' + str(characteristic) + '.png'
    x = np.array(LANDMARKS_COUNT)
    bins = np.array(LANDMARKS_COUNT)
    y1 = []
    y2 = []
    y3 = []
    y4 = []

    for count in LANDMARKS_COUNT:
        if len(marks_selection) == 1:
            y1.append(results[method][marks_selection[0]][count][characteristic])
        elif len(marks_selection) == 2:
            y1.append(results[method][marks_selection[0]][count][characteristic])
            y2.append(results[method][marks_selection[1]][count][characteristic])
        else:
            y1.append(results[method][marks_selection[0]][count][characteristic])
            y2.append(results[method][marks_selection[1]][count][characteristic])
            y3.append(results[method][marks_selection[2]][count][characteristic])
        y4.append(results['BFS'])


    fig, ax = plt.subplots()
    if len(marks_selection) == 1:
        plt.plot(x, y1, '-bo', label=str(marks_selection[0]))
    elif len(marks_selection) == 2:
        plt.plot(x, y1, '-bo', label=str(marks_selection[0]))
        plt.plot(x, y2, '-ro', label=str(marks_selection[1]))
    else:
        plt.plot(x, y1, '-bo', label=str(marks_selection[0]))
        plt.plot(x, y2, '-ro', label=str(marks_selection[1]))
        plt.plot(x, y3, '-go', label=str(marks_selection[2]))
    if method != 'Landmarks selection' and method != 'Finding pathes':
        plt.plot(x, y4, '-mo', label='BFS (as standard)')
    plt.xticks(bins)
    ax.legend(fontsize = 14)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.xlabel('Number of Landmarks', fontsize = 20)
    plt.ylabel(str(characteristic), fontsize = 20)
    plt.title(str(method), fontsize=30)
    fig.set_figwidth(12)    #  ширина Figure
    fig.set_figheight(7)    #  высота Figure
    fig.savefig(name)
#------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#основная часть
import random
#import networkx as nx
import collections

total_time = tm.datetime.now()

marks_selection = ['random', 'degree', 'coverege']
#estimation_strategies = ('basic', 'landmarks_bfs')

results = {'Basic':{
                    'random': {}, 
                    'degree': {},
                    'coverege': {}
                 },
        'Landmarks-BFS': {
            'random': {}, 
            'degree': {},
            'coverege': {}
            },
        'BFS': 0,
        'Landmarks selection': {
            'random': {}, 
            'degree': {},
            'coverege': {}
            },
        'Finding pathes': {
            'random': {},
            'degree': {},
            'coverege': {}
            }
         }
if type(LANDMARKS_COUNT) == int:
    LANDMARKS_COUNT = [LANDMARKS_COUNT]
for i in LANDMARKS_COUNT:
    for j in marks_selection:
        results['Basic'][j][i] = {'Time, sec': 0, 'Approximation error': 0}
        results['Landmarks-BFS'][j][i] = {'Time, sec': 0, 'Approximation error': 0}
        results['Landmarks selection'][j][i] = {'Time, sec': 0}
        results['Finding pathes'][j][i] = {'Time, sec': 0}

with open('results_vk.txt', 'w') as file:
    
    graph = read_from_txt(FILENAME)
    number_of_tests = 1

    #для тестов с заданными вершинами
    #test(file, results, graph, """количество тестов""" 1, """start""" 1, """finish""" 200, """способ выбора марок""" ['random', 'degree', 'coverege'])

    marks_selection = ['degree', 'coverege']
    test(file, results, graph, 1, 1, 200, marks_selection)

    #test(file, results, graph, number_of_tests) #простой запуск (все варианты для случайных start и finish)

total_time = tm.datetime.now() - total_time
print('Total time: ', total_time)
create_plot(results, 'Basic', 'Time, sec', marks_selection)
create_plot(results, 'Landmarks-BFS', 'Time, sec', marks_selection)
create_plot(results, 'Landmarks selection', 'Time, sec', marks_selection)
create_plot(results, 'Finding pathes', 'Time, sec', marks_selection)
create_histogram(results, 'Basic', 'Approximation error', marks_selection)
create_histogram(results, 'Landmarks-BFS', 'Approximation error', marks_selection)
plt.show()
