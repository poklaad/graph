FILENAME = '3.txt'
ORIENTED = False
LANDMARKS_COUNT = range(20, 101, 40)
#LANDMARKS_COUNT = range(2, 10, 2)
import datetime as tm
import networkx as nx







 





 
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#считывание данных
import csv
from operator import truediv

def read_from_txt(filename, oriented=False):
    graph = {}
    with open(filename) as file:

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
                    'length': {}, #length можно убрать, тк ее можно получить как len(shortest_paths[v])
                    'shortest_paths':{},
                    'degree': 1,
                }

            if oriented:
                if child not in graph:
                    graph[child] = {
                        'linked': [],
                        'length': {},
                        'shortest_paths':{},
                        'degree': 0,
                    }

            else:
                if child in graph:
                    if parent not in graph[child]['linked']:
                        graph[child]['linked'].append(parent)
                        graph[child]['degree'] += 1

                else:
                    graph[child] = {
                        'linked': [parent],
                        'length': {},
                        'shortest_paths':{},
                        'degree': 1,
                    }

            row = file.readline()

    return graph

def read_from_csv(filename, oriented=False):
    graph = {}

    with open(filename) as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:

            parent = int(row[0])
            child = int(row[1])

            if parent in graph:
                if child not in graph[parent]['linked']:
                    graph[parent]['linked'].append(child)
                    graph[parent]['degree'] += 1
            else:
                graph[parent] = {
                    'linked': [child],
                    'length': {},
                    'shortest_paths':{},
                    'degree': 1,
                }

            if oriented:
                if child not in graph:
                    graph[child] = {
                        'linked': [],
                        'length': {},
                        'shortest_paths':{},
                        'degree': 0,
                    }

            else:
                if child in graph:
                    if parent not in graph[child]['linked']:
                        graph[child]['linked'].append(parent)
                        graph[child]['degree'] += 1

                else:
                    graph[child] = {
                        'linked': [parent],
                        'length': {},
                        'shortest_paths':{},
                        'degree': 1,
                    }

    return graph

def parse(filename, oriented=False):
    if filename.split('.')[0] == 'vk':
        return read_from_csv(filename, oriented)
    elif filename.split('.')[-1] == 'txt':
        return read_from_txt(filename, oriented)













#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#вычисление кратчайших путей между двумя вершинами
#это просто bfs, только на выход дает маршрут
def short_path (start, finish, graph):
    from collections import deque 
    found = False
    visited = []
    visited.append(start)
    queue = deque()
    queue.append(start)
    pathes = {}
    pathes [start]=[start]
    while queue:
        v = queue.popleft() 
        if v == finish:
            found = True
            path = pathes[v]
            
            break

        for neighbor in graph[v]['linked']:
            if neighbor not in visited:
                visited.append(neighbor)
                queue.append(neighbor)
                pathes[neighbor] = []
                pathes[neighbor] = pathes[v].copy()
                pathes[neighbor].append(neighbor)


    if found:
        return path
    else:
        return -1













#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#выбор марок
import random


def landmarks_choose(start, finish, graph, selection_type, landmarks_count):
    start_time= tm.datetime.now()
    graph_items = graph.items()
    graph_size = len(graph_items)

    
    

    if selection_type == 'random':
        rand_nod = [i[0] for i in graph_items]
        random.shuffle(rand_nod)
        rand_nod = rand_nod[:landmarks_count]
        ###############################################################################################################################################
        #rand_nod = [7, 5, 6, 10, 11, 13, 2, 3]
        ###############################################################################################################################################
        for v in rand_nod:
            path_from_start = short_path(start, v, graph)
            graph[start]['shortest_paths'][v] = path_from_start
            if path_from_start != -1:
                graph[start]['length'][v] = len(path_from_start)#length можно убрать, тк ее можно получить как len(shortest_paths[v])
            else: 
                graph[start]['length'][v] = -1#length можно убрать, тк ее можно получить как len(shortest_paths[v])

            path_from_finish = short_path(v, finish, graph)
            graph[finish]['shortest_paths'][v] = path_from_finish
            if path_from_finish != -1:
                graph[finish]['length'][v] = len(path_from_finish)#length можно убрать, тк ее можно получить как len(shortest_paths[v])
            else:
                graph[finish]['length'][v] = -1#length можно убрать, тк ее можно получить как len(shortest_paths[v])
        return rand_nod, (tm.datetime.now() - start_time).total_seconds()



    elif selection_type == 'degree':
        sort_degree = sorted(graph_items, key=lambda x: x[1]['degree'], reverse=True) 
        degree_nod = [i[0] for i in sort_degree[:landmarks_count]]
        ###############################################################################################################################################
        #degree_nod = [7, 5, 6, 10, 11, 13, 2, 3]
        ###############################################################################################################################################

        for v in degree_nod:
            path_from_start = short_path(start, v, graph)
            graph[start]['shortest_paths'][v] = path_from_start
            if path_from_start != -1:
                graph[start]['length'][v] = len(path_from_start)#length можно убрать, тк ее можно получить как len(shortest_paths[v])
            else: 
                graph[start]['length'][v] = -1#length можно убрать, тк ее можно получить как len(shortest_paths[v])

            path_from_finish = short_path(v, finish, graph)
            graph[finish]['shortest_paths'][v] = path_from_finish
            if path_from_finish != -1:
                graph[finish]['length'][v] = len(path_from_finish)#length можно убрать, тк ее можно получить как len(shortest_paths[v])
            else:
                graph[finish]['length'][v] = -1#length можно убрать, тк ее можно получить как len(shortest_paths[v])

        return degree_nod, (tm.datetime.now() - start_time).total_seconds()




    elif selection_type == 'coverege':
        number_of_uses = {}
        while len(number_of_uses) < landmarks_count:
            uses_part = {}
            rand_nod = [i[0] for i in graph_items]
            random.shuffle(rand_nod)
            start_nod = rand_nod[:landmarks_count]
            finish_nod = rand_nod[graph_size - landmarks_count:]
            for i, j in zip(start_nod, finish_nod):
                ##used_nodes - список с номерами вершин, которые попали в кратчайший путь
                used_nodes = short_path (i,j, graph)
                if used_nodes == -1:
                    continue
                for v in used_nodes:
                    if v not in uses_part:
                        uses_part[v] = {#можно просто число хранить, без словаря с count
                            'count' : 1
                        }
                    else:
                        uses_part[v]['count'] += 1#можно просто число хранить, без словаря с count
            number_of_uses = {**number_of_uses, **uses_part}

        number_of_uses = sorted(number_of_uses, reverse=True)
        number_of_uses = number_of_uses[:landmarks_count]




        ###############################################################################################################################################
        #number_of_uses = [7, 5, 6, 10, 11, 13, 2, 3]
        ###############################################################################################################################################
        for v in number_of_uses:
            path_from_start = short_path(start, v, graph)
            graph[start]['shortest_paths'][v] = path_from_start
            if path_from_start != -1:
                graph[start]['length'][v] = len(path_from_start)#length можно убрать, тк ее можно получить как len(shortest_paths[v])
            else: 
                graph[start]['length'][v] = -1#length можно убрать, тк ее можно получить как len(shortest_paths[v])

            path_from_finish = short_path(v, finish, graph)
            graph[finish]['shortest_paths'][v] = path_from_finish
            if path_from_finish != -1:
                graph[finish]['length'][v] = len(path_from_finish)#length можно убрать, тк ее можно получить как len(shortest_paths[v])
            else:
                graph[finish]['length'][v] = -1#length можно убрать, тк ее можно получить как len(shortest_paths[v])
        return number_of_uses, (tm.datetime.now() - start_time).total_seconds()

        """берется M случайных пар вершин
        вычисляется кратчайший путь между ними
        для каждой вершины в этом пути увеличивается количество ее вхождений в кратчайшие пути
        выбираются вершины с самым большим количеством вхождений
        """















 
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#bfs возвращает расстояние между двумя вершинами
def bfs(start, finish, graph):
    start_time= tm.datetime.now()
    if len(graph) == 0 or start not in graph or finish not in graph:
        return -1, (tm.datetime.now() - start_time).total_seconds()
    from collections import deque 
    found = False
    visited = []
    visited.append(start)
    queue = deque()
    queue.append(start)
    pathes = {}
    pathes [start]={'shortest_path' : [start]}#можно убрать shortest_path
    
    while queue:
        v = queue.popleft() 
        if v == finish:
            found = True
            path = pathes[v]['shortest_path']
            break

        for neighbor in graph[v]['linked']:
            if neighbor not in visited:
                visited.append(neighbor)
                queue.append(neighbor)
                p = pathes[v]['shortest_path'].copy()#можно убрать посредника р и сразу заночить в  pathes[neighbor]
                p.append(neighbor)
                pathes[neighbor] = {'shortest_path' : []}
                pathes[neighbor]['shortest_path'] = p


    if found:
        return path, (tm.datetime.now() - start_time).total_seconds()
    else:
        return -1, (tm.datetime.now() - start_time).total_seconds()












 
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#bfs+landmarks
def landmarks_bfs(start, finish, graph):
    start_time= tm.datetime.now()
    little_graph = {}
    start_path = graph[start]['shortest_paths']
    finish_path = graph[finish]['shortest_paths']
    


    for mark, path in list(start_path.items()) + list(finish_path.items()):
        if path == -1 or path == []:
            continue
        if path[len(path)-1] not in little_graph:
#            if path[len(path)-1] == 0.0:
 #               a=5
            little_graph[path[len(path)-1]] = { 
                'linked': [],
            }
        for v in range(len(path)-1):
            if path[v] in little_graph and path[v+1] not in little_graph[path[v]]['linked']:
                little_graph[path[v]]['linked'].append(path[v+1])
            elif path[v] not in little_graph:
#                if path[v] == 0.0:
 #                   a=5
                little_graph[path[v]] = {
                    'linked': [path[v+1]],
                }


            if path[v+1] in little_graph and path[v] not in little_graph[path[v+1]]['linked']:
                little_graph[path[v+1]]['linked'].append(path[v])
            elif path[v+1] not in little_graph:
#                if path[v+1] == 0.0:
 #                   a=5
                little_graph[path[v+1]] = {
                    'linked': [path[v]],
                }
                
            

    #fl = little_graph[start]
    path, bfs_time = bfs(start, finish, little_graph)
    return path, (tm.datetime.now() - start_time).total_seconds()













 
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#basic+landmarks
def landmarks_basic(start, finish, graph):
    start_time= tm.datetime.now()
    start_path = graph[start]['shortest_paths']
    finish_path = graph[finish]['shortest_paths']
    shortest_start_path = []
    shortest_finish_path = []

    
    d_up = 10**6
    
    for mark, from_start in start_path.items():
        
        to_finish = finish_path.get(mark, -1)
        if to_finish == -1 or to_finish == [] or from_start == [] or from_start == -1:
            continue
        to_finish.pop(0)
        d = len(from_start) + len(to_finish)
        
        if d < d_up:
            d_up = d
            shortest_start_path = from_start
            shortest_finish_path = to_finish

            
    #проверка на достижимость
    if d_up == 10**6:
        return -1, (tm.datetime.now() - start_time).total_seconds()
    shortest_path = shortest_start_path + shortest_finish_path
    return shortest_path, (tm.datetime.now() - start_time).total_seconds()












 
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#основная часть
import random
import networkx as nx
import collections


"""[здесь будет словарь с ключом - количество лендмарок и значениями: точность и время работы]"""
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
        'bfs': {
            'random': {}, 
            'degree': {},
            'coverege': {}
            },
        'choose_landmarks': {
            'random': {}, 
            'degree': {},
            'coverege': {}
            }
         }
for landmarks_count in LANDMARKS_COUNT:
    results['basic']['random'][landmarks_count] = {'time': 0, 'accuracy': 0}
                                  
    results['basic']['degree'][landmarks_count] = {'time': 0,'accuracy': 0}
                                  
    results['basic']['coverege'][landmarks_count] = {'time': 0,'accuracy': 0}
                                  
    results['landmarks_bfs']['random'][landmarks_count] = {'time': 0, 'accuracy': 0}
                                  
    results['landmarks_bfs']['degree'][landmarks_count] = {'time': 0,'accuracy': 0}
                                  
    results['landmarks_bfs']['coverege'][landmarks_count] = {'time': 0,'accuracy': 0}

    results['bfs']['random'][landmarks_count] = {'time': 0}
                                  
    results['bfs']['degree'][landmarks_count] = {'time': 0}
                                  
    results['bfs']['coverege'][landmarks_count] = {'time': 0}
                                  
    results['choose_landmarks']['random'][landmarks_count] = {'time': 0}
                                  
    results['choose_landmarks']['degree'][landmarks_count] = {'time': 0}
                                  
    results['choose_landmarks']['coverege'][landmarks_count] = {'time': 0}


with open('results_test_CA-GrQc.txt', 'w') as file:
    tic = tm.datetime.now()
    #    file.write('Start time: ' + '\n')
    #   file.write(str(tic) + '\n')

    graph = parse(FILENAME, ORIENTED)
    graph_items = graph.items()
    graph_size = len(graph_items)
    marks_selection = ('random', 'degree', 'coverege')








    #    file.write('Count of nodes: ' + str(graph_size) + '\n')




    #выбор начальной и конечной вершины

    number_of_tests = 100     # сделать больше тестов



    nodes = [i[0] for i in graph_items]
    nodes_start = nodes.copy()
    random.shuffle(nodes_start)
    nodes_start = nodes_start[:number_of_tests]

    nodes_finish = nodes.copy()
    random.shuffle(nodes_finish)
    nodes_finish = nodes_finish[:number_of_tests]

    #запуск алгоритмов с разными параметрами




    real_bfs_difference = {}
    real_bfs_difference[-1] = 0 #landmarks_bfs did not find the way
    real_bfs_difference[-2] = 0 #нет пути между вершинами
    approx_bfs_difference = {}
    approx_bfs_difference[-1] = 0 #landmarks_bfs did not find the way
    approx_bfs_difference[-2] = 0 #landmarks_basic did not find the way
    approx_bfs_difference[-3] = 0 #landmarks_basic and landmarks_bfs did not find the way
    approx_bfs_difference[-4] = 0 #нет пути между вершинами



    #    G = nx.Graph()
    #   with open('3.txt') as graph_file:
    #      row = graph_file.readline()
    #     while row:
    #        parent, child = row.split()
        #       parent = (int(parent))
        #      child = (int(child))
        #     if parent not in G:
        #        G.add_node(parent)
            #   if child not in G:
            #      G.add_node(child)
            # if (parent, child) not in G:
            #    G.add_edge(parent, child)
    #            if (child, parent) not in G:
    #               G.add_edge(child, parent)
    #          row = graph_file.readline()




    
    for selection in marks_selection:
        for landmarks_count in LANDMARKS_COUNT:
            for start, finish in zip(nodes_start,nodes_finish):

            #сброс данных
    #            for k, v in graph.items():
    #               v['length'] = {}
    #              v['shortest_paths'] = {}
            
                for k, v in graph.items():
                    v['length'] = {}
                    v['shortest_paths'] = {}
                while start == finish:
                    if start < graph_size-1:
                        start += 1
                    else:
                        start -= 1
                ###############################################################################################################################################
                #start = 8
                #finish = 9
                ###############################################################################################################################################

    #                file.write('-' * 50 + '\n')
    #               file.write(selection + '\n') 
    #              file.write('-' * 50 + '\n')
    #             file.write('landmarks_count: ' + str(landmarks_count) + '\n')
    #            file.write('-' * 50 + '\n')
        #           file.write('Start: ' + str(start) + '\n') 
        #          file.write('Finish: ' + str(finish) + '\n') 
        #         file.write('=' * 50 + '\n')




                landmarks, timer_landmarks = landmarks_choose (start, finish, graph, selection, landmarks_count)
                results['choose_landmarks'][selection][landmarks_count]['time'] += timer_landmarks
                #print(timer_landmarks)
                

    #                file.write ('Landmarks: ' + str(landmarks) + '\n')
    #               file.write('=' * 50 + '\n')



                path_bfs, timer_bfs = landmarks_bfs (start, finish, graph)
                #print (results['landmarks_bfs'][selection][landmarks_count]['time'])
                #print (timer_bfs)
                results['landmarks_bfs'][selection][landmarks_count]['time'] += timer_bfs



                path_basic, timer_basic = landmarks_basic (start, finish, graph)
                results['basic'][selection][landmarks_count]['time'] += timer_basic


                s_path, timer_exact = bfs(start, finish, graph)
                results['bfs'][selection][landmarks_count]['time'] += timer_exact
 


    #                results_time['basic'][selection][landmarks_count] = {timer_bfs, timer_basic, """"точность и время алгоритмов"""}
    #               results_time['landmarks_bfs'][selection][landmarks_count] = {timer_bfs, timer_basic, """"точность и время алгоритмов"""}
    #              results_time['bfs'][selection][landmarks_count] = {timer_bfs, timer_basic, """"точность и время алгоритмов"""}

                


    #                if path_bfs != -1:
    #                   file.write('Distance landmarks_bfs:' + str(len(path_bfs)) + '\n')
    #              else:
    #                 file.write('Distance landmarks_bfs:' + str(-1) + '\n')
    #            if path_basic != -1:
        #               file.write('Distance path_basic:' + str(len(path_basic)) + '\n')
        #          else:
        #             file.write('Distance path_basic:' + str(-1) + '\n')
        #        if s_path != -1:
            #           file.write('Distance path_real:' + str(len(s_path)) + '\n')
            #      else:
            #         file.write('Distance path_real:' + str(-1) + '\n')
            #    file.write('=' * 50 + '\n')



    #                file.write('Path for pure bfs:' + str(s_path) + '\n')
    #               file.write('=' * 50 + '\n')

    #              file.write('Path for landmarks_bfs:' + str(path_bfs) + '\n')
    #             file.write('=' * 50 + '\n')
 
    #            file.write('Path for landmarks_basic: ' + str(path_basic) + '\n')
        #           file.write('=' * 50 + '\n')






    #                try:
    #                   s_path = nx.shortest_path(G, source=start, target=finish, weight=None, method='dijkstra')
    #              except Exception:
    #                 s_path = -1
    #            file.write('Real shortest path:' + str(s_path) + '\n')
        #           if s_path != -1:
        #              file.write('Distance real shortest path: ' + str(len(s_path)) + '\n')
        #         else:
        #            file.write('Distance real shortest path: ' + str(-1) + '\n')
            #       file.write('=' * 50 + '\n')
            
#                if s_path == -1 and path_bfs == -1:
#                    difference = -2
                if s_path != -1 and path_bfs != -1:
#                    difference = len(path_bfs) - len(s_path)
                    approximation_error_bfs = (len(path_bfs) - len(s_path))/len(s_path)
                    results['landmarks_bfs'][selection][landmarks_count]['accuracy'] += approximation_error_bfs
#                else:
#                    difference = -1
                
                


#                if difference not in real_bfs_difference:
 #                   real_bfs_difference[difference] = 1
  #              else:
   #                 real_bfs_difference[difference] += 1



#                if s_path == -1:
 #                   accurateness = -4
  #              elif s_path != -1 and path_bfs == -1 and path_basic != -1:
   #                 accurateness = -1
    #            elif s_path != -1 and path_bfs != -1 and path_basic == -1:
     #               accurateness = -2
      #          elif s_path != -1 and path_bfs == -1 and path_basic == -1:
       #             accurateness = -3
        #        else:
                if s_path != -1 and path_bfs != -1 and path_basic != -1:
#                    accurateness = len(path_basic) - len(path_bfs)
                    approximation_error_basic = (len(path_basic) - len(s_path))/len(s_path)
                    results['basic'][selection][landmarks_count]['accuracy'] += approximation_error_basic

                
#                if accurateness not in approx_bfs_difference:
 #                   approx_bfs_difference[accurateness] = 1
  #              else:
   #                 approx_bfs_difference[accurateness] += 1


                """operations = operations_basic - operations_bfs
                time = time_basic.total_seconds() - time_bfs.total_seconds()"""

    #                file.write('accurateness: ' + str(accurateness) + '\n')
                """print('Operations: ' + str(operations))
                print('Time: ' + str(time))"""
    #               file.write('*' * 100 + '\n')

            results['basic'][selection][landmarks_count]['accuracy'] /= number_of_tests
            results['landmarks_bfs'][selection][landmarks_count]['accuracy'] /= number_of_tests
            results['choose_landmarks'][selection][landmarks_count]['time'] /= number_of_tests
            results['landmarks_bfs'][selection][landmarks_count]['time'] /= number_of_tests
            results['basic'][selection][landmarks_count]['time'] /= number_of_tests
            results['bfs'][selection][landmarks_count]['time'] /= number_of_tests
            





    #    file.write('The number of cases when the landmarks_bfs did not find the way (real_bfs_difference[-1])' + '\n')
    #   file.write('The number of cases when there is no way between vertices (real_bfs_difference[-2])' + '\n')
    #  file.write('Number of differences between exact shortest path and landmarks_bfs shortest path: ' + '\n')

#    real_bfs_difference = collections.OrderedDict(sorted(real_bfs_difference.items()))

    #    file.write('{')
    #   for difference in real_bfs_difference:
    #      file.write(str(difference) + ': ' + str(real_bfs_difference[difference]) + '\n')
    # file.write('}' + '\n')




    #    file.write('The number of cases when the landmarks_bfs did not find the way (approx_bfs_difference[-1])' + '\n')
    #   file.write('The number of cases when the landmarks_basic did not find the way (approx_bfs_difference[-2])' + '\n')
    #  file.write('The number of cases when the landmarks_basic and landmarks_bfs did not find the way (approx_bfs_difference[-3])' + '\n')
    # file.write('The number of cases when there is no way between vertices (approx_bfs_difference[-4])' + '\n')
    #file.write('Number of differences between landmarks_basic shortest path and landmarks_bfs shortest path: ' + '\n')

#    approx_bfs_difference = collections.OrderedDict(sorted(approx_bfs_difference.items()))

    #    file.write('{')
    #   for accurateness in approx_bfs_difference:
    #      file.write(str(accurateness) + ': ' + str(approx_bfs_difference[accurateness]) + '\n')
    # file.write('}' + '\n')



    toc = tm.datetime.now()
    #    file.write('Finish time: ' + '\n')
    #   file.write(str(toc) + '\n')
    #  file.write('Working time: ' + '\n')
    file.write(str(toc - tic) + '\n')



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
    y_bfs_random_time.append(results['bfs']['random'][count]['time'])
    y_bfs_degree_time.append(results['bfs']['degree'][count]['time'])
    y_bfs_coverege_time.append(results['bfs']['coverege'][count]['time'])###########################################################
    y_choose_landmarks_random_time.append(results['choose_landmarks']['random'][count]['time'])
    y_choose_landmarks_degree_time.append(results['choose_landmarks']['degree'][count]['time'])
    y_choose_landmarks_coverege_time.append(results['choose_landmarks']['coverege'][count]['time'])###########################################################

    y_basic_random_accuracy.append(results['basic']['random'][count]['accuracy'])
    y_basic_degree_accuracy.append(results['basic']['degree'][count]['accuracy'])
    y_basic_coverege_accuracy.append(results['basic']['coverege'][count]['accuracy'])###########################################################
    y_landmarks_bfs_random_accuracy.append(results['landmarks_bfs']['random'][count]['accuracy'])
    y_landmarks_bfs_degree_accuracy.append(results['landmarks_bfs']['degree'][count]['accuracy'])
    y_landmarks_bfs_coverege_accuracy.append(results['landmarks_bfs']['coverege'][count]['accuracy'])###########################################################




#    y_bfs_random_accuracy.append(results['bfs']['random'][count]['accuracy'])
#    y_bfs_degree_accuracy.append(results['bfs']['degree'][count]['accuracy'])
#    y_bfs_coverege_accuracy.append(results['bfs']['coverege'][count]['time'])###########################################################
#    y_choose_landmarks_random_accuracy.append(results['choose_landmarks']['random'][count]['accuracy'])
#    y_choose_landmarks_degree_accuracy.append(results['choose_landmarks']['degree'][count]['accuracy'])
#    y_choose_landmarks_coverege_accuracy.append(results['choose_landmarks']['coverege'][count]['accuracy'])###########################################################


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


#fig, ax = plt.subplots()
#plt.plot(x, y_bfs_random_accuracy, color='blue')
#plt.plot(x, y_bfs_degree_accuracy, color='red')
#plt.plot(x, y_bfs_coverege_accuracy, color='green')


#fig, ax = plt.subplots()
#plt.plot(x, y_choose_landmarks_random_accuracy, color='blue')
#plt.plot(x, y_choose_landmarks_degree_accuracy, color='red')
#plt.plot(x, y_choose_landmarks_coverege_accuracy, color='green')





#plt.xlabel('Percent of landmarks')
#plt.ylabel('Operations delta (operations number)')
#plt.title('Operations delta with differend strategies of landmarks selection using middle point distance estimation', fontsize=30)






plt.show()