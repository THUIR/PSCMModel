#!/usr/bin/env python
#coding: utf-8


import sys
import gc
import json
import math

from collections import defaultdict, namedtuple
from datetime import datetime

from wc_common import *


from config_sample import *
    
from POM_function import *
    


REL_PRIORS = (0.5, 0.5)

DEFAULT_REL = REL_PRIORS[1] / sum(REL_PRIORS)

MAX_QUERY_ID = 1000     # some initial value that will be updated by InputReader

item_base_info_list = ['intentWeight', 'query', 'urls', 'layout', 'clicks', 'click_times']

SessionItem = namedtuple('SessionItem', item_base_info_list)

class ClickModel:

    def __init__(self, ignoreIntents=True, ignoreLayout=True):
        self.ignoreIntents = ignoreIntents
        self.ignoreLayout = ignoreLayout
        
    def output_perplexity(self, file_name):
        #query  session_count overall_perplexity
        #perplexity for postion 1 to 10
        out_file = open(file_name, "w")
        out_file.write(str("Overall") + "\t" + str(self.test_perplexity_session_count) + "\t" + str(self.test_perplexity_position[MAX_DOCS_PER_QUERY]) + "\n")
        out_file.write(arr_string(self.test_perplexity_position[0:MAX_DOCS_PER_QUERY]) + "\n")
        for q in xrange(MAX_QUERY_ID):
            out_file.write(str(q) + "\t" + str(self.test_perplexity_query_session_count[q]) + "\t" + str(self.test_perplexity_query_position[q][MAX_DOCS_PER_QUERY]) + "\n")
            out_file.write(arr_string(self.test_perplexity_query_position[q][0:MAX_DOCS_PER_QUERY]) + "\n")
        out_file.close()
        
    def train(self, sessions):
        """
            Set some attributes that will be further used in _getClickProbs function
        """
        pass

    def test(self, sessions, reportPositionPerplexity=True):
        self.test_perplexity_position = [0.0 for i in xrange(MAX_DOCS_PER_QUERY + 1)] 
        self.test_perplexity_query_position = [[0.0 for i in xrange(MAX_DOCS_PER_QUERY + 1)] for q in xrange(MAX_QUERY_ID)] # [0]: value, [1]: count, last_element: overall
        self.test_perplexity_query_session_count = [0 for q in xrange(MAX_QUERY_ID)]
        self.test_perplexity_session_count = 0
        logLikelihood = 0.0
        positionPerplexity = [0.0] * MAX_DOCS_PER_QUERY
        positionPerplexityClickSkip = [[0.0, 0.0] for i in xrange(MAX_DOCS_PER_QUERY)]
        counts = [0] * MAX_DOCS_PER_QUERY
        countsClickSkip = [[0, 0] for i in xrange(MAX_DOCS_PER_QUERY)]
        possibleIntents = [False] if self.ignoreIntents else [False, True]
        process_count_tmp = 0
        for s in sessions:
            process_count_tmp += 1
            #if process_count_tmp % 100 == 0:
            #    print "test count "+ str(process_count_tmp)
            query = s.query
            iw = s.intentWeight
            intentWeight = {False: 1.0} if self.ignoreIntents else {False: 1 - iw, True: iw}
            clickProbs = self._getClickProbs(s, possibleIntents)
            N = len(s.clicks)
            if DEBUG:
                assert N > 1
                x = sum(clickProbs[i][N // 2] * intentWeight[i] for i in possibleIntents) / sum(clickProbs[i][N // 2 - 1] * intentWeight[i] for i in possibleIntents)
                s.clicks[N // 2] = 1 if s.clicks[N // 2] == 0 else 0
                clickProbs2 = self._getClickProbs(s, possibleIntents)
                y = sum(clickProbs2[i][N // 2] * intentWeight[i] for i in possibleIntents) / sum(clickProbs2[i][N // 2 - 1] * intentWeight[i] for i in possibleIntents)
                assert abs(x + y - 1) < 0.01, (x, y)
            logLikelihood += math.log(sum(clickProbs[i][N - 1] * intentWeight[i] for i in possibleIntents))      # log_e
            correctedRank = 0    # we are going to skip clicks on fake pager urls
            self.test_perplexity_query_session_count[query] += 1
            self.test_perplexity_session_count += 1
            for k, click in enumerate(s.clicks):
                click = 1 if click else 0
                # P(C_k | C_1, ..., C_{k-1}) = \sum_I P(C_1, ..., C_k | I) P(I) / \sum_I P(C_1, ..., C_{k-1} | I) P(I)
                curClick = dict((i, clickProbs[i][k]) for i in possibleIntents)
                prevClick = dict((i, clickProbs[i][k - 1]) for i in possibleIntents) if k > 0 else dict((i, 1.0) for i in possibleIntents)
                logProb = math.log(sum(curClick[i] * intentWeight[i] for i in possibleIntents), 2) - math.log(sum(prevClick[i] * intentWeight[i] for i in possibleIntents), 2)
                positionPerplexity[correctedRank] += logProb
                positionPerplexityClickSkip[correctedRank][click] += logProb
                counts[correctedRank] += 1
                countsClickSkip[correctedRank][click] += 1
                self.test_perplexity_query_position[query][correctedRank] += logProb
                self.test_perplexity_query_position[query][MAX_DOCS_PER_QUERY] += logProb
                correctedRank += 1
        positionPerplexity = [2 ** (-x / count if count else x) for (x, count) in zip(positionPerplexity, counts)]
        positionPerplexityClickSkip = [[2 ** (-x[click] / (count[click] if count[click] else 1) if count else x) \
                for (x, count) in zip(positionPerplexityClickSkip, countsClickSkip)] for click in xrange(2)]
        perplexity = sum(positionPerplexity) / len(positionPerplexity)
        N = len(sessions)
        
        #print str(positionPerplexityClickSkip)
        
        for q in xrange(MAX_QUERY_ID):
            for i in range(0, MAX_DOCS_PER_QUERY + 1):
                _x = self.test_perplexity_query_position[q][i]
                _c = self.test_perplexity_query_session_count[q]
                if i == MAX_DOCS_PER_QUERY:
                    _c *= MAX_DOCS_PER_QUERY
                self.test_perplexity_query_position[q][i] = (2.0 ** (- _x / _c if _c else _x))
        for i in range(0, MAX_DOCS_PER_QUERY):
            self.test_perplexity_position[i] = positionPerplexity[i]
        self.test_perplexity_position[MAX_DOCS_PER_QUERY] = perplexity
        
        ret_str = "LogLikelihood\t" + str(logLikelihood / N / MAX_DOCS_PER_QUERY) + "\n"
        ret_str += "Perplexity\t" + str(perplexity) + "\n"
        ret_str += "positionPerplexity"
        for i in range(0, MAX_DOCS_PER_QUERY):
            ret_str += "\t" + str(positionPerplexity[i])
        ret_str += "\n"
        
        ret_str += "positionPerplexitySkip"
        for i in range(0, MAX_DOCS_PER_QUERY):
            ret_str += "\t" + str(positionPerplexityClickSkip[0][i])
        ret_str += "\n"
        
        ret_str += "positionPerplexityClick"
        for i in range(0, MAX_DOCS_PER_QUERY):
            ret_str += "\t" + str(positionPerplexityClickSkip[1][i])
        ret_str += "\n"
        
        return ret_str

    def _getClickProbs(self, s, possibleIntents):
        """
            Returns clickProbs list
            clickProbs[i][k] = P(C_1, ..., C_k | I=i)
        """
        return dict((i, [0.5 ** (k + 1) for k in xrange(len(s.clicks))]) for i in possibleIntents)


class DbnModel(ClickModel):

    def __init__(self, gammas, ignoreIntents=True, ignoreLayout=True):
        self.gammas = gammas
        ClickModel.__init__(self, ignoreIntents, ignoreLayout)
    
    def get_model_info(self):
        ret = "+++++DBN:\n"
        return ret
    
    def get_relevance_list(self):
        possibleIntents = [False] if self.ignoreIntents else [False, True]
        ret = []
        for q in xrange(MAX_QUERY_ID):
            for url in self.urlRelevances[False][q].keys():
                ret.append([q, url, self.urlRelevances[False][q][url]['a'] * self.urlRelevances[False][q][url]['s']])
        return ret

    def train(self, sessions):
        possibleIntents = [False] if self.ignoreIntents else [False, True]
        # intent -> query -> url -> (a_u, s_u)
        self.urlRelevances = dict((i, [defaultdict(lambda: {'a': DEFAULT_REL, 's': DEFAULT_REL}) for q in xrange(MAX_QUERY_ID)]) for i in possibleIntents)
        # here we store distribution of posterior intent weights given train data
        self.queryIntentsWeights = defaultdict(lambda: [])

        # EM algorithm
        if not PRETTY_LOG:
            print >>sys.stderr, '-' * 80
            print >>sys.stderr, 'Start. Current time is', datetime.now()
        for iteration_count in xrange(MAX_ITERATIONS):
            # urlRelFractions[intent][query][url][r][1] --- coefficient before \log r
            # urlRelFractions[intent][query][url][r][0] --- coefficient before \log (1 - r)
            urlRelFractions = dict((i, [defaultdict(lambda: {'a': [1.0, 1.0], 's': [1.0, 1.0]}) for q in xrange(MAX_QUERY_ID)]) for i in [False, True])
            self.queryIntentsWeights = defaultdict(lambda: [])
            # E step
            for s in sessions:
                positionRelevances = {}
                query = s.query
                for intent in possibleIntents:
                    positionRelevances[intent] = {}
                    for r in ['a', 's']:
                        positionRelevances[intent][r] = [self.urlRelevances[intent][query][url][r] for url in s.urls]
                layout = [False] * len(s.layout) if self.ignoreLayout else s.layout
                sessionEstimate = dict((intent, self._getSessionEstimate(positionRelevances[intent], layout, s.clicks, intent)) for intent in possibleIntents)

                # P(I | C, G)
                if self.ignoreIntents:
                    p_I__C_G = {False: 1, True: 0}
                else:
                    a = sessionEstimate[False]['C'] * (1 - s.intentWeight)
                    b = sessionEstimate[True]['C'] * s.intentWeight
                    p_I__C_G = {False: a / (a + b), True: b / (a + b)}
                self.queryIntentsWeights[query].append(p_I__C_G[True])
                for k, url in enumerate(s.urls):
                    for intent in possibleIntents:
                        # update a
                        urlRelFractions[intent][query][url]['a'][1] += sessionEstimate[intent]['a'][k] * p_I__C_G[intent]
                        urlRelFractions[intent][query][url]['a'][0] += (1 - sessionEstimate[intent]['a'][k]) * p_I__C_G[intent]
                        if s.clicks[k] != 0:
                            # Update s
                            urlRelFractions[intent][query][url]['s'][1] += sessionEstimate[intent]['s'][k] * p_I__C_G[intent]
                            urlRelFractions[intent][query][url]['s'][0] += (1 - sessionEstimate[intent]['s'][k]) * p_I__C_G[intent]
            if not PRETTY_LOG:
                sys.stderr.write('E')

            # M step
            # update parameters and record mean square error
            sum_square_displacement = 0.0
            Q_functional = 0.0
            num_points = 0
            for i in possibleIntents:
                for query, d in enumerate(urlRelFractions[i]):
                    if not d:
                        continue
                    for url, relFractions in d.iteritems():
                        a_u_new = relFractions['a'][1] / (relFractions['a'][1] + relFractions['a'][0])
                        sum_square_displacement += (a_u_new - self.urlRelevances[i][query][url]['a']) ** 2
                        num_points += 1
                        self.urlRelevances[i][query][url]['a'] = a_u_new
                        Q_functional += relFractions['a'][1] * math.log(a_u_new) + relFractions['a'][0] * math.log(1 - a_u_new)
                        s_u_new = relFractions['s'][1] / (relFractions['s'][1] + relFractions['s'][0])
                        sum_square_displacement += (s_u_new - self.urlRelevances[i][query][url]['s']) ** 2
                        num_points += 1
                        self.urlRelevances[i][query][url]['s'] = s_u_new
                        Q_functional += relFractions['s'][1] * math.log(s_u_new) + relFractions['s'][0] * math.log(1 - s_u_new)
            if not PRETTY_LOG:
                sys.stderr.write('M\n')
            rmsd = math.sqrt(sum_square_displacement / (num_points if TRAIN_FOR_METRIC else 1.0))
            if PRETTY_LOG:
                sys.stderr.write('%d..' % (iteration_count + 1))
            else:
                print >>sys.stderr, 'Iteration: %d, RMSD: %.10f' % (iteration_count + 1, rmsd)
                print >>sys.stderr, 'Q functional: %f' % Q_functional
            del urlRelFractions
        if PRETTY_LOG:
            sys.stderr.write('\n')

    @staticmethod
    def testBackwardForward():
        positionRelevances = {'a': [0.5] * MAX_DOCS_PER_QUERY, 's': [0.5] * MAX_DOCS_PER_QUERY}
        gammas = [0.9] * 4
        layout = [False] * (MAX_DOCS_PER_QUERY + 1)
        clicks = [0] * MAX_DOCS_PER_QUERY
        alpha, beta = DbnModel.getForwardBackwardEstimates(positionRelevances, gammas, layout, clicks, False)
        x = alpha[0][0] * beta[0][0] + alpha[0][1] * beta[0][1]
        assert all(abs((a[0] * b[0] + a[1] * b[1]) / x  - 1) < 0.00001 for a, b in zip(alpha, beta))

    @staticmethod
    def getGamma(gammas, k, layout, intent):
        index = 2 * (1 if layout[k + 1] else 0) + (1 if intent else 0)
        return gammas[index]

    @staticmethod
    def getForwardBackwardEstimates(positionRelevances, gammas, layout, clicks, intent):
        N = len(clicks)
        if DEBUG:
            assert N + 1 == len(layout)
        alpha = [[0.0, 0.0] for i in xrange(N + 1)]
        beta = [[0.0, 0.0] for i in xrange(N + 1)]
        alpha[0] = [0.0, 1.0]
        beta[N] = [1.0, 1.0]

        # P(E_{k+1} = e, C_k | E_k = e', G, I)
        updateMatrix = [[[0.0 for e1 in [0, 1]] for e in [0, 1]] for i in xrange(N)]
        for k, C_k in enumerate(clicks):
            a_u = positionRelevances['a'][k]
            s_u = positionRelevances['s'][k]
            gamma = DbnModel.getGamma(gammas, k, layout, intent)
            if C_k == 0:
                updateMatrix[k][0][0] = 1
                updateMatrix[k][0][1] = (1 - gamma) * (1 - a_u)
                updateMatrix[k][1][0] = 0
                updateMatrix[k][1][1] = gamma * (1 - a_u)
            else:
                updateMatrix[k][0][0] = 0
                updateMatrix[k][0][1] = (s_u + (1 - gamma) * (1 - s_u)) * a_u
                updateMatrix[k][1][0] = 0
                updateMatrix[k][1][1] = gamma * (1 - s_u) * a_u

        for k in xrange(N):
            for e in [0, 1]:
                alpha[k + 1][e] = sum(alpha[k][e1] * updateMatrix[k][e][e1] for e1 in [0, 1])
                beta[N - 1 - k][e] = sum(beta[N - k][e1] * updateMatrix[N - 1 - k][e1][e] for e1 in [0, 1])

        return alpha, beta

    def _getSessionEstimate(self, positionRelevances, layout, clicks, intent):
        # Returns {'a': P(A_k | I, C, G), 's': P(S_k | I, C, G), 'C': P(C | I, G), 'clicks': P(C_k | C_1, ..., C_{k-1}, I, G)} as a dict
        # sessionEstimate[True]['a'][k] = P(A_k = 1 | I = 'Fresh', C, G), probability of A_k = 0 can be calculated as 1 - p
        N = len(clicks)
        if DEBUG:
            assert N + 1 == len(layout)
        sessionEstimate = {'a': [0.0] * N, 's': [0.0] * N, 'e': [[0.0, 0.0] for k in xrange(N)], 'C': 0.0, 'clicks': [0.0] * N}

        alpha, beta = self.getForwardBackwardEstimates(positionRelevances, self.gammas, layout, clicks, intent)
        try:
            varphi = [((a[0] * b[0]) / (a[0] * b[0] + a[1] * b[1]), (a[1] * b[1]) / (a[0] * b[0] + a[1] * b[1])) for a, b in zip(alpha, beta)]
        except ZeroDivisionError:
            print >>sys.stderr, alpha, beta, [(a[0] * b[0] + a[1] * b[1]) for a, b in zip(alpha, beta)], positionRelevances
            sys.exit(1)
        if DEBUG:
            assert all(ph[0] < 0.01 for ph, c in zip(varphi[:N], clicks) if c != 0), (alpha, beta, varphi, clicks)
        # calculate P(C | I, G) for k = 0
        sessionEstimate['C'] = alpha[0][0] * beta[0][0] + alpha[0][1] * beta[0][1]      # == 0 + 1 * beta[0][1]
        for k, C_k in enumerate(clicks):
            a_u = positionRelevances['a'][k]
            s_u = positionRelevances['s'][k]
            gamma = self.getGamma(self.gammas, k, layout, intent)
            # E_k_multiplier --- P(S_k = 0 | C_k) P(C_k | E_k = 1)
            if C_k == 0:
                sessionEstimate['a'][k] = a_u * varphi[k][0]
                sessionEstimate['s'][k] = 0.0
            else:
                sessionEstimate['a'][k] = 1.0
                sessionEstimate['s'][k] = varphi[k + 1][0] * s_u / (s_u + (1 - gamma) * (1 - s_u))
            # P(C_1, ..., C_k | I)
            sessionEstimate['clicks'][k] = sum(alpha[k + 1])

        return sessionEstimate

    def _getClickProbs(self, s, possibleIntents):
        """
            Returns clickProbs list:
            clickProbs[i][k] = P(C_1, ..., C_k | I=i)
        """
        # TODO: ensure that s.clicks[l] not used to calculate clickProbs[i][k] for l >= k
        positionRelevances = {}
        for intent in possibleIntents:
            positionRelevances[intent] = {}
            for r in ['a', 's']:
                positionRelevances[intent][r] = [self.urlRelevances[intent][s.query][url][r] for url in s.urls]
                if QUERY_INDEPENDENT_PAGER:
                    for k, u in enumerate(s.urls):
                        if u == 'PAGER':
                            # use dummy 0 query for all fake pager URLs
                            positionRelevances[intent][r][k] = self.urlRelevances[intent][0][url][r]
        layout = [False] * len(s.layout) if self.ignoreLayout else s.layout
        return dict((i, self._getSessionEstimate(positionRelevances[i], layout, s.clicks, i)['clicks']) for i in possibleIntents)


class UbmModel(ClickModel):

    gammaTypesNum = 4

    def __init__(self, ignoreIntents=True, ignoreLayout=True, explorationBias=False):
        self.explorationBias = explorationBias
        ClickModel.__init__(self, ignoreIntents, ignoreLayout)
        
    def train(self, sessions):
        possibleIntents = [False] if self.ignoreIntents else [False, True]
        # alpha: intent -> query -> url -> "attractiveness probability"
        self.alpha = dict((i, [defaultdict(lambda: DEFAULT_REL) for q in xrange(MAX_QUERY_ID)]) for i in possibleIntents)
        # gamma: freshness of the current result: gammaType -> rank -> "distance from the last click" - 1 -> examination probability
        self.gamma = [[[0.5 for d in xrange(MAX_DOCS_PER_QUERY)] for r in xrange(MAX_DOCS_PER_QUERY)] for g in xrange(self.gammaTypesNum)]
        if self.explorationBias:
            self.e = [0.5 for p in xrange(MAX_DOCS_PER_QUERY)]
        if not PRETTY_LOG:
            print >>sys.stderr, '-' * 80
            print >>sys.stderr, 'Start. Current time is', datetime.now()
        for iteration_count in xrange(MAX_ITERATIONS):
            self.queryIntentsWeights = defaultdict(lambda: [])
            # not like in DBN! xxxFractions[0] is a numerator while xxxFraction[1] is a denominator
            alphaFractions = dict((i, [defaultdict(lambda: [1.0, 2.0]) for q in xrange(MAX_QUERY_ID)]) for i in possibleIntents)
            gammaFractions = [[[[1.0, 2.0] for d in xrange(MAX_DOCS_PER_QUERY)] for r in xrange(MAX_DOCS_PER_QUERY)] for g in xrange(self.gammaTypesNum)]
            if self.explorationBias:
                eFractions = [[1.0, 2.0] for p in xrange(MAX_DOCS_PER_QUERY)]
            # E-step
            for s in sessions:
                query = s.query
                layout = [False] * len(s.layout) if self.ignoreLayout else s.layout
                if self.explorationBias:
                    explorationBiasPossible = any((l and c for (l, c) in zip(s.layout, s.clicks)))
                    firstVerticalPos = -1 if not any(s.layout[:-1]) else [k for (k, l) in enumerate(s.layout) if l][0]
                if self.ignoreIntents:
                    p_I__C_G = {False: 1.0, True: 0}
                else:
                    a = self._getSessionProb(s) * (1 - s.intentWeight)
                    b = 1 * s.intentWeight
                    p_I__C_G = {False: a / (a + b), True: b / (a + b)}
                self.queryIntentsWeights[query].append(p_I__C_G[True])
                prevClick = -1
                for rank, c in enumerate(s.clicks):
                    url = s.urls[rank]
                    for intent in possibleIntents:
                        a = self.alpha[intent][query][url]
                        if self.explorationBias and explorationBiasPossible:
                            e = self.e[firstVerticalPos]
                        if c == 0:
                            g = self.getGamma(self.gamma, rank, prevClick, layout, intent)
                            gCorrection = 1
                            if self.explorationBias and explorationBiasPossible and not s.layout[k]:
                                gCorrection = 1 - e
                                g *= gCorrection
                            alphaFractions[intent][query][url][0] += a * (1 - g) / (1 - a * g) * p_I__C_G[intent]
                            self.getGamma(gammaFractions, rank, prevClick, layout, intent)[0] += g / gCorrection * (1 - a) / (1 - a * g) * p_I__C_G[intent]
                            if self.explorationBias and explorationBiasPossible:
                                eFractions[firstVerticalPos][0] += (e if s.layout[k] else e / (1 - a * g)) * p_I__C_G[intent]
                        else:
                            alphaFractions[intent][query][url][0] += 1 * p_I__C_G[intent]
                            self.getGamma(gammaFractions, rank, prevClick, layout, intent)[0] += 1 * p_I__C_G[intent]
                            if self.explorationBias and explorationBiasPossible:
                                eFractions[firstVerticalPos][0] += (e if s.layout[k] else 0) * p_I__C_G[intent]
                        alphaFractions[intent][query][url][1] += 1 * p_I__C_G[intent]
                        self.getGamma(gammaFractions, rank, prevClick, layout, intent)[1] += 1 * p_I__C_G[intent]
                        if self.explorationBias and explorationBiasPossible:
                            eFractions[firstVerticalPos][1] += 1 * p_I__C_G[intent]
                    if c != 0:
                        prevClick = rank
            if not PRETTY_LOG:
                sys.stderr.write('E')
            # M-step
            sum_square_displacement = 0.0
            num_points = 0
            for i in possibleIntents:
                for q in xrange(MAX_QUERY_ID):
                    for url, aF in alphaFractions[i][q].iteritems():
                        new_alpha = aF[0] / aF[1]
                        sum_square_displacement += (self.alpha[i][q][url] - new_alpha) ** 2
                        num_points += 1
                        self.alpha[i][q][url] = new_alpha
            for g in xrange(self.gammaTypesNum):
                for r in xrange(MAX_DOCS_PER_QUERY):
                    for d in xrange(MAX_DOCS_PER_QUERY):
                        gF = gammaFractions[g][r][d]
                        new_gamma = gF[0] / gF[1]
                        sum_square_displacement += (self.gamma[g][r][d] - new_gamma) ** 2
                        num_points += 1
                        self.gamma[g][r][d] = new_gamma
            if self.explorationBias:
                for p in xrange(MAX_DOCS_PER_QUERY):
                    new_e = eFractions[p][0] / eFractions[p][1]
                    sum_square_displacement += (self.e[p] - new_e) ** 2
                    num_points += 1
                    self.e[p] = new_e
            if not PRETTY_LOG:
                sys.stderr.write('M\n')
            rmsd = math.sqrt(sum_square_displacement / (num_points if TRAIN_FOR_METRIC else 1.0))
            if PRETTY_LOG:
                sys.stderr.write('%d..' % (iteration_count + 1))
            else:
                print >>sys.stderr, 'Iteration: %d, RMSD: %.10f' % (iteration_count + 1, rmsd)
        if PRETTY_LOG:
            sys.stderr.write('\n')
        for q, intentWeights in self.queryIntentsWeights.iteritems():
            self.queryIntentsWeights[q] = sum(intentWeights) / len(intentWeights)

    def _getSessionProb(self, s):
        clickProbs = self._getClickProbs(s, [False, True])
        N = len(s.clicks)
        return clickProbs[False][N - 1] / clickProbs[True][N - 1]

    @staticmethod
    def getGamma(gammas, k, prevClick, layout, intent):
        index = (2 if layout[k] else 0) + (1 if intent else 0)
        return gammas[index][k][k - prevClick - 1]

    def _getClickProbs(self, s, possibleIntents):
        """
            Returns clickProbs list
            clickProbs[i][k] = P(C_1, ..., C_k | I=i)
        """
        clickProbs = dict((i, []) for i in possibleIntents)
        firstVerticalPos = -1 if not any(s.layout[:-1]) else [k for (k, l) in enumerate(s.layout) if l][0]
        prevClick = -1
        layout = [False] * len(s.layout) if self.ignoreLayout else s.layout
        for rank, c in enumerate(s.clicks):
            url = s.urls[rank]
            prob = {False: 0.0, True: 0.0}
            for i in possibleIntents:
                a = self.alpha[i][s.query][url]
                g = self.getGamma(self.gamma, rank, prevClick, layout, i)
                if self.explorationBias and any(s.layout[k] and s.clicks[k] for k in xrange(rank)) and not s.layout[rank]:
                    g *= 1 - self.e[firstVerticalPos]
                prevProb = 1 if rank == 0 else clickProbs[i][-1]
                if c == 0:
                    clickProbs[i].append(prevProb * (1 - a * g))
                else:
                    clickProbs[i].append(prevProb * a * g)
            if c != 0:
                prevClick = rank
        return clickProbs


class InputReader:
    def __init__(self, discardNoClicks=True):
        self.url_to_id = {}
        self.query_to_id = {}
        self.current_url_id = 1
        self.current_query_id = 0
        self.discardNoClicks = discardNoClicks

    def __call__(self, f, query_class_map):
        sessions = []
        #
        session_count = 0
        in_file = open(f)
        while True:
            line = in_file.readline()
            if not line:
                break
            arr = line.rstrip().split('\t')
            if len(arr) < 10:
                continue
            query = int(arr[0])
            urls = string_arr(arr[1], " ", "int")
            clicks = string_arr(arr[2], " ", "int")
            click_times = string_arr(arr[3], " ", "float")
            exams = {}
            layout = []
            for i in range(0, MAX_DOCS_PER_QUERY):
                layout.append(False)
            region = 1
            intentWeight = 1.0
            extra = {}
            urlsObserved = 0
            urls = urls[:MAX_DOCS_PER_QUERY]
            urlsObserved = len(urls)
            layout = layout[:urlsObserved]
            clicks = clicks[:urlsObserved]
            click_times = click_times[:urlsObserved]
            if urlsObserved < MIN_DOCS_PER_QUERY:
                continue
            intentWeight = float(intentWeight)
            layout.append(False)
            sessions.append(SessionItem(intentWeight, query, urls, layout, clicks, click_times))
            session_count += 1
        in_file.close()
        print "from " + f + " load " + str(session_count) + " sessions"
        # FIXME: bad style
        #global MAX_QUERY_ID
        #MAX_QUERY_ID = wc_max_id
        return sessions

    @staticmethod
    def convertToList(sparseDict, defaultElem=0, maxLen=MAX_DOCS_PER_QUERY):
        """ Convert dict of the format {"0": doc0, "13": doc13} to the list of the length MAX_DOCS_PER_QUERY """
        convertedList = [defaultElem] * maxLen
        extra = {}
        for k, v in sparseDict.iteritems():
            try:
                convertedList[int(k)] = v
            except (ValueError, IndexError):
                extra[k] = v
        return convertedList, extra

class NaiveModel(ClickModel):

    def __init__(self, ignoreExamInProb=True, ignoreExamInCTR=True, ignoreIntents=True, ignoreLayout=True):
        self.ignoreExamInProb = ignoreExamInProb
        self.ignoreExamInCTR = ignoreExamInCTR
        ClickModel.__init__(self, ignoreIntents, ignoreLayout)
    
    gammaTypesNum = 4

    def train(self, sessions):
        possibleIntents = [False] if self.ignoreIntents else [False, True]
        urlRelFractions = dict((i, [defaultdict(lambda: [1.0, 1.0]) for q in xrange(MAX_QUERY_ID)]) for i in possibleIntents)
        print('session = ' + str(len(sessions)))
        for s in sessions:
            query = s.query
            layout = [False] * len(s.layout) if self.ignoreLayout else s.layout
            intentWeights = {False: 1.0} if self.ignoreIntents else {False: 1 - s.intentWeight, True: s.intentWeight}
            #for k, (u, c) in enumerate(zip(s.urls, s.clicks[:(lastClickedPos + 1)])):
            for k, (u, c) in enumerate(zip(s.urls, s.clicks)):
                for i in possibleIntents:
                    if c != 0:
                        urlRelFractions[i][query][u][1] += intentWeights[i]
                    else:
                        exam = 1.0
                        urlRelFractions[i][query][u][0] += intentWeights[i]*exam
        self.urlRelevances = dict((i, [defaultdict(lambda: DEFAULT_REL) for q in xrange(MAX_QUERY_ID)]) for i in possibleIntents)
        for i in possibleIntents:
            for query, d in enumerate(urlRelFractions[i]):
                if not d:
                    continue
                for url, relFractions in d.iteritems():
                    #print(str(url) + " : " + str(relFractions[0]))
                    self.urlRelevances[i][query][url] = relFractions[1] / (relFractions[1] + relFractions[0])
        #print("relevance" + str(self.urlRelevances[False]))

    def _getClickProbs(self, s, possibleIntents):
        clickProbs = {False: [], True: []}          # P(C_1, ..., C_k)
        query = s.query
        layout = [False] * len(s.layout) if self.ignoreLayout else s.layout
        for i in possibleIntents:
            examinationProb = 1.0       # P(C_1, ..., C_{k - 1}, E_k = 1)
            for k, c in enumerate(s.clicks):
                r = self.urlRelevances[i][query][s.urls[k]]
                prevProb = 1 if k == 0 else clickProbs[i][-1]
                exam = 1
                if c == 0:
                    clickProbs[i].append(prevProb * (1 - exam * r))    # P(C_1, ..., C_k = 0) = P(C_1, ..., C_{k-1}) - P(C_1, ..., C_k = 1)
                else:
                    clickProbs[i].append(prevProb * exam * r)
        for i in possibleIntents:
            for j in range(0,len(clickProbs[i])):
                if clickProbs[i][j] <= 0:
                    clickProbs[i][j] = 0.00000000000000000000001
        return clickProbs

    @staticmethod
    def getGamma(gammas, k, layout, intent):
        return DbnModel.getGamma(gammas, k, layout, intent)
    
    def getRelevance(self, query_url_set, readInput):
        rel_set = {}
        count = 0
        for query in query_url_set:
            try:
                q_id = readInput.query_to_id[(query,readInput.region)]
                rel_set[query] = {}
                for url in query_url_set[query]:
                    u_id = readInput.url_to_id[url]
                    if self.urlRelevances[False][q_id].has_key(u_id):
                        rel_set[query][url] = self.urlRelevances[False][q_id][u_id]
            except:
                continue
        #print('match ' + str(count) + ' ' + str(len(rel_set)))
        return rel_set
    
    # def getRelSet(self):
        # rel_set = {}
        # for q in xrange(len(self.urlRelevances[False])):
            # rel_set[q] = {}
            # for u in self.urlRelevances[False][q]:
                # rel_set[q][u] = self.urlRelevances[False][q][url]
        # return rel_set 
    
    def get_model_info(self):
        ret = "+++++NaiveModel:\n"
        return ret
        
    def get_relevance_list(self):
        ret = []
        for q in xrange(MAX_QUERY_ID):
            for url in self.urlRelevances[False][q]:
                ret.append([q, url, self.urlRelevances[False][q][url]])
        return ret


class WCRealUbmModel(ClickModel):
    #gammaTypesNum = 1
    def __init__(self, CLASS_K, ignoreIntents=True, ignoreLayout=True, explorationBias=False):
        self.explorationBias = explorationBias
        ClickModel.__init__(self, ignoreIntents, ignoreLayout)
        self.gammaTypesNum = CLASS_K
    
    def get_model_info(self):
        ret = "+++++Gamma:\n"
        for m in range(0, self.gammaTypesNum):
            ret += "M : " + str(m) + "\n"
            for r in xrange(MAX_DOCS_PER_QUERY):
                for d in xrange(MAX_DOCS_PER_QUERY):
                    ret += str(self.gamma[m][r][d]) + "\t"
                ret += "\n"
        return ret
        
    def get_relevance_list(self):
        ret = []
        for q in xrange(MAX_QUERY_ID):
            for url in self.alpha[q].keys():
                ret.append([q, url, self.alpha[q][url]])
        return ret
    
    def train(self, sessions):
        #possibleIntents = [False] if self.ignoreIntents else [False, True]# no use
        # alpha: intent -> query -> url -> "attractiveness probability"
        self.alpha = [defaultdict(lambda: DEFAULT_REL) for q in xrange(MAX_QUERY_ID)]
        self.mu = [[(1.0 / self.gammaTypesNum) for g in xrange(self.gammaTypesNum)] for q in xrange(MAX_QUERY_ID)]
        # gamma: freshness of the current result: gammaType -> rank -> "distance from the last click" - 1 -> examination probability
        self.gamma = [[[0.5 for d in xrange(MAX_DOCS_PER_QUERY)] for r in xrange(MAX_DOCS_PER_QUERY)] for g in xrange(self.gammaTypesNum)]
        if not PRETTY_LOG:
            print >>sys.stderr, '-' * 80
            print >>sys.stderr, 'Start. Current time is', datetime.now()
        for iteration_count in xrange(MAX_ITERATIONS):
            self.queryIntentsWeights = defaultdict(lambda: [])
            # not like in DBN! xxxFractions[0] is a numerator while xxxFraction[1] is a denominator
            alphaFractions = [defaultdict(lambda: [1.0, 2.0]) for q in xrange(MAX_QUERY_ID)]
            muFranctions = [[[0.0, 0.0] for g in xrange(self.gammaTypesNum)] for q in xrange(MAX_QUERY_ID)]
            gammaFractions = [[[[1.0, 2.0] for d in xrange(MAX_DOCS_PER_QUERY)] for r in xrange(MAX_DOCS_PER_QUERY)] for g in xrange(self.gammaTypesNum)]
            
            # E-step
            for s in sessions:
                query = s.query
                for m in range(0, self.gammaTypesNum):
                    muFranctions[query][m][1] += 1
                layout = [False] * len(s.layout) if self.ignoreLayout else s.layout
                p_I__C_G = {False: 1.0, True: 0}
                self.queryIntentsWeights[query].append(p_I__C_G[True])
                prevClick = -1
                for rank, c in enumerate(s.clicks):
                    url = s.urls[rank]
                    a = self.alpha[query][url]
                    r = rank
                    d = rank - prevClick - 1
                    g = self.getSumGamma(s, rank, prevClick)
                    #if g == 0.0:
                    #    print self.mu
                    if c == 0:
                        alphaFractions[query][url][0] += a * (1 - g) / (1 - a * g)
                        for m in range(0, self.gammaTypesNum):
                            gammaFractions[m][r][d][0] += self.gamma[m][r][d] * self.mu[query][m] * (1.0 - a) / (1.0 - a * g)
                            gammaFractions[m][r][d][1] += self.mu[query][m] * (1.0 - a * self.gamma[m][r][d]) / (1.0 - a * g)
                            muFranctions[query][m][0] += self.mu[query][m] * (1.0 - a * self.gamma[m][r][d]) / (1.0 - a * g)
                    else:
                        alphaFractions[query][url][0] += 1
                        for m in range(0, self.gammaTypesNum):
                            gammaFractions[m][r][d][0] += 1
                            gammaFractions[m][r][d][1] += 1
                            muFranctions[query][m][0] += self.gamma[m][r][d] * self.mu[query][m] / g
                    alphaFractions[query][url][1] += 1
                    if c != 0:
                        prevClick = rank
            if not PRETTY_LOG:
                sys.stderr.write('E')
            # M-step
            sum_square_displacement = 0.0
            num_points = 0
            for q in xrange(MAX_QUERY_ID):
                mu_sum = 0.0
                for m in range(0, self.gammaTypesNum):
                    if muFranctions[q][m][1] > 0:
                        new_mu = muFranctions[q][m][0] / muFranctions[q][m][1]
                        sum_square_displacement += (self.mu[q][m] - new_mu) ** 2
                        num_points += 1
                        self.mu[q][m] = new_mu
                        mu_sum += self.mu[q][m]
                if mu_sum == 0.0:
                    for m in range(0, self.gammaTypesNum):
                        self.mu[q][m] = 1.0 / self.gammaTypesNum
                else:
                    for m in range(0, self.gammaTypesNum):
                        self.mu[q][m] /= mu_sum
            for q in xrange(MAX_QUERY_ID):
                for url, aF in alphaFractions[q].iteritems():
                    new_alpha = aF[0] / aF[1]
                    sum_square_displacement += (self.alpha[q][url] - new_alpha) ** 2
                    num_points += 1
                    self.alpha[q][url] = new_alpha
            for g in xrange(self.gammaTypesNum):
                for r in xrange(MAX_DOCS_PER_QUERY):
                    for d in xrange(MAX_DOCS_PER_QUERY):
                        gF = gammaFractions[g][r][d]
                        new_gamma = gF[0] / gF[1]
                        sum_square_displacement += (self.gamma[g][r][d] - new_gamma) ** 2
                        num_points += 1
                        self.gamma[g][r][d] = new_gamma
            if not PRETTY_LOG:
                sys.stderr.write('M\n')
            rmsd = math.sqrt(sum_square_displacement / (num_points if TRAIN_FOR_METRIC else 1.0))
            if PRETTY_LOG:
                sys.stderr.write('%d..' % (iteration_count + 1))
            else:
                print >>sys.stderr, 'Iteration: %d, RMSD: %.10f' % (iteration_count + 1, rmsd)
            del alphaFractions
            del gammaFractions
            del muFranctions
        if PRETTY_LOG:
            sys.stderr.write('\n')

    def _getSessionProb(self, s):
        clickProbs = self._getClickProbs(s, [False, True])
        N = len(s.clicks)
        return clickProbs[False][N - 1] / clickProbs[True][N - 1]
    
    def getSumGamma(self, session, k, prevClick):
        q = session.query
        ret = 0.0
        for m in range(0, self.gammaTypesNum):
            ret += (self.mu[q][m] * self.gamma[m][k][k - prevClick - 1])
        return ret
        
        
    # @staticmethod
    # def getGamma(gammas, k, prevClick, layout, intent):
        # index = (2 if layout[k] else 0) + (1 if intent else 0)
        # return gammas[index][k][k - prevClick - 1]
    # @staticmethod
    # def getWCGamma(gammas, k, prevClick):
        # return gammas[0][k][k - prevClick - 1]
    
    # @staticmethod
    # def getWCGamma(gammas, k, prevClick, m):
        # return gammas[m][k][k - prevClick - 1]

    def _getClickProbs(self, s, possibleIntents):
        """
            Returns clickProbs list
            clickProbs[i][k] = P(C_1, ..., C_k | I=i)
        """
        clickProbs = dict((i, []) for i in possibleIntents)
        firstVerticalPos = -1 if not any(s.layout[:-1]) else [k for (k, l) in enumerate(s.layout) if l][0]
        prevClick = -1
        layout = [False] * len(s.layout) if self.ignoreLayout else s.layout
        for rank, c in enumerate(s.clicks):
            url = s.urls[rank]
            prob = {False: 0.0, True: 0.0}
            for i in possibleIntents:
                a = self.alpha[s.query][url]
                g = self.getSumGamma(s, rank, prevClick)
                #g = self.getWCGamma(self.gamma, rank, prevClick)
                prevProb = 1 if rank == 0 else clickProbs[i][-1]
                if c == 0:
                    clickProbs[i].append(prevProb * (1 - a * g))
                else:
                    clickProbs[i].append(prevProb * a * g)
            if c != 0:
                prevClick = rank
        return clickProbs

class POMModel(ClickModel):        
    def __init__(self, ignoreIntents=True, ignoreLayout=True, explorationBias=False, train_round = 10):
        self.train_round = train_round
        self.explorationBias = explorationBias
        ClickModel.__init__(self, ignoreIntents, ignoreLayout)
    
    def generate_click_seq(self, click_list, click_time_list):
        click_seq_list = []
        for i in range(0, len(click_list)):
            if click_list[i] == 1:
                click_seq_list.append([i, click_time_list[i]])
        click_seq_list = sorted(click_seq_list, key = lambda x: x[1], reverse=False)
        return click_seq_list
    
    def get_model_info(self):
        ret = "+++++V:\n"
        for m in xrange(MAX_DOCS_PER_QUERY):
            for n in xrange(MAX_DOCS_PER_QUERY):
                ret += str(self.query_V[m][n]) + "\t"
            ret += "\n"
        return ret
        
    def get_relevance_list(self):
        ret = []
        for q in xrange(MAX_QUERY_ID):
            for url in self.alpha[q].keys():
                ret.append([q, url, self.alpha[q][url]])
        return ret
    
    def get_s_list(self, s):
        S_list = []
        for url in s.urls:
            S_list.append(1.0 - self.alpha[s.query][url])
        return S_list
    
    def train(self, sessions):
        #possibleIntents = [False] if self.ignoreIntents else [False, True]# no use
        self.alpha = [defaultdict(lambda: DEFAULT_REL) for q in xrange(MAX_QUERY_ID)]
        self.query_V = [[(1.0 / (MAX_DOCS_PER_QUERY + 1)) for j in xrange(MAX_DOCS_PER_QUERY + 1)] for i in xrange(MAX_DOCS_PER_QUERY + 1)]
        #self.query_S = [[1.0 for i in xrange(MAX_DOCS_PER_QUERY)] for q in xrange(MAX_QUERY_ID)]
        self.first_click_prob = [[1.0 for i in xrange(MAX_DOCS_PER_QUERY + 1)] for q in xrange(MAX_QUERY_ID)]

        for iteration_count in xrange(MAX_ITERATION_POM):
            iter_V_up = [[0.0 for j in xrange(MAX_DOCS_PER_QUERY + 1)] for i in xrange(MAX_DOCS_PER_QUERY + 1)]
            iter_V_down = [0.0 for i in xrange(MAX_DOCS_PER_QUERY + 1)]
            iter_S_up = [defaultdict(lambda: 0) for q in xrange(MAX_QUERY_ID)]
            iter_S_down = [defaultdict(lambda: 0) for q in xrange(MAX_QUERY_ID)]
            
            session_p_count = 0
            for s in sessions:
                # session_p_count += 1
                # if session_p_count % 100 == 0:
                    # print "POM process " + str(session_p_count)
                query = s.query
                ori_v_list = [MAX_DOCS_PER_QUERY]
                click_seq_list = self.generate_click_seq(s.clicks, s.click_times)
                for i in range(0, len(click_seq_list)):
                    ori_v_list.append(click_seq_list[i][0])
                ori_v_list.append(MAX_DOCS_PER_QUERY)
                ori_s_list = [0 for x in ori_v_list]
                path_list = []
                S_list = self.get_s_list(s)
                #print "C-" + str(ori_v_list)
                #print "S-" + str(ori_s_list)
                add_Qk_list(0, len(ori_v_list) - 1, path_list, ori_v_list, ori_s_list, self.query_V, S_list, self.first_click_prob[query], iteration_count, MAX_QK_LENGTH, MAX_INSERT_NUM, MAX_DOCS_PER_QUERY, MAX_TOP_N, 1)
                if len(path_list) > 50:
                    print "path list num " + str(len(path_list))
                if len(path_list) == 0:
                    #ignore no click session
                    continue
                for path in path_list:
                    for i in range(0, len(path.v_list) - 1):
                        m = path.v_list[i]
                        n = path.v_list[i + 1]
                        iter_V_up[m][n] = iter_V_up[m][n] + path.prob
                    for i in range(0, len(path.v_list)):
                        m = path.v_list[i]
                        if not m == MAX_DOCS_PER_QUERY:
                            url_m = s.urls[m]
                            iter_V_down[m] = iter_V_down[m] + path.prob
                            iter_S_down[query][url_m] = iter_S_down[query][url_m] + path.prob
                            if path.s_list[i] == 1:
                                iter_S_up[query][url_m] = iter_S_up[query][url_m] + path.prob
            #print "----\n" + str(iter_V_down)
            for m in range(0, MAX_DOCS_PER_QUERY + 1):
                if iter_V_down[m] > 0:
                    for n in range(0, MAX_DOCS_PER_QUERY + 1):
                        self.query_V[m][n] = iter_V_up[m][n] / iter_V_down[m]
            for query in range(0, MAX_QUERY_ID):
                for url in iter_S_down[query].keys():
                    if iter_S_down[query][url] > 0:
                        self.alpha[query][url] = 1.0 - (iter_S_up[query][url] / iter_S_down[query][url])
            del iter_V_up
            del iter_V_down
            del iter_S_up
            del iter_S_down               
            if not PRETTY_LOG:
                sys.stderr.write('M\n')
            if PRETTY_LOG:
                sys.stderr.write('%d..' % (iteration_count + 1))
            else:
                print >>sys.stderr, 'Iteration: %d' % (iteration_count + 1)
                
    def _getClickProbs(self, s, possibleIntents):
        """
            Returns clickProbs list
            clickProbs[i][k] = P(C_1, ..., C_k | I=i)
        """
        clickProbs = dict((poi, []) for poi in possibleIntents)
        layout = [False] * len(s.layout) if self.ignoreLayout else s.layout
        query = s.query
        S_list = self.get_s_list(s)
        P0T = [[1.0, 0.0, 0.0, 0.0] for k in xrange(MAX_DOCS_PER_QUERY)]
        #print str(self.query_V)
        #print str(S_list)
        for m in range(0, MAX_DOCS_PER_QUERY):
            P0T[m][2] = self.query_V[MAX_DOCS_PER_QUERY][m]
        for v in range(0, MAX_POM_CHAIN):
            for m in range(0, MAX_DOCS_PER_QUERY):
                P0T[m][0] *= (1.0 - P0T[m][2] * (1.0 - max(min(S_list[m], 0.999999), 0.0000001)))
            for m in range(0, MAX_DOCS_PER_QUERY):
                P0T[m][3] = 0
                for n in range(0, MAX_DOCS_PER_QUERY):
                    P0T[m][3] += P0T[n][2] * self.query_V[n][m]
            for m in range(0, MAX_DOCS_PER_QUERY):
                P0T[m][2] = max(min(P0T[m][3], 0.999999), 0.0000001)
            
        #print str(P0T)
        for rank, c in enumerate(s.clicks):
            url = s.urls[rank]
            prob = {False: 0.0, True: 0.0}
            for poi in possibleIntents:
                prevProb = 1 if rank == 0 else clickProbs[poi][-1]
                if c == 0:
                    clickProbs[poi].append(prevProb * P0T[rank][0])
                else:
                    clickProbs[poi].append(prevProb * (1.0 - P0T[rank][0]))
        return clickProbs
        
class RevisitModelUBM(ClickModel):        
    def __init__(self, ignoreIntents=True, ignoreLayout=True, explorationBias=False):
        self.explorationBias = explorationBias
        ClickModel.__init__(self, ignoreIntents, ignoreLayout)
    
    def get_model_info(self):
        ret = "+++++Gamma:\n"
        for i in range(MAX_DOCS_PER_QUERY):
            ret += "i : " + str(i) + "\n"
            for m in xrange(MAX_DOCS_PER_QUERY + 1):
                for n in xrange(MAX_DOCS_PER_QUERY + 1):
                    ret += str(self.gamma[i][m][n]) + "\t"
                ret += "\n"
        return ret
        
    def get_relevance_list(self):
        ret = []
        for q in xrange(MAX_QUERY_ID):
            for url in self.alpha[q].keys():
                ret.append([q, url, self.alpha[q][url]])
        return ret
    
    def generate_click_seq(self, click_list, click_time_list):
        click_seq_list = []
        for i in range(0, len(click_list)):
            if click_list[i] == 1:
                click_seq_list.append([i, click_time_list[i]])
        click_seq_list = sorted(click_seq_list, key = lambda x: x[1], reverse=False)
        return click_seq_list
        
    def train(self, sessions):
        #possibleIntents = [False] if self.ignoreIntents else [False, True]# no use
        # alpha: attractiveness probability[ q: query, u:url]
        self.alpha = [defaultdict(lambda: DEFAULT_REL) for q in xrange(MAX_QUERY_ID)]
        #self.mu = [[(1.0 / self.gammaTypesNum) for g in xrange(self.gammaTypesNum)] for q in xrange(MAX_QUERY_ID)]
        #gamma: examination probability[ i current rank, m: prev click (first is -1 (MAX_DOCS_PER_QUERY in array)), n:next click (last no click is MAX_DOCS_PER_QUERY)]
        self.gamma = [[[0.5 for n in xrange(MAX_DOCS_PER_QUERY + 1)] for m in xrange(MAX_DOCS_PER_QUERY + 1)] for i in xrange(MAX_DOCS_PER_QUERY)]
        if not PRETTY_LOG:
            print >>sys.stderr, '-' * 80
            print >>sys.stderr, 'Start. Current time is', datetime.now()
        for iteration_count in xrange(MAX_ITERATIONS):
            self.queryIntentsWeights = defaultdict(lambda: [])
            # not like in DBN! xxxFractions[0] is a numerator while xxxFraction[1] is a denominator
            alphaFractions = [defaultdict(lambda: [1.0, 0.5, 0.5]) for q in xrange(MAX_QUERY_ID)]#part (a) (b) (c)
            #muFranctions = [[[0.0, 0.0] for g in xrange(self.gammaTypesNum)] for q in xrange(MAX_QUERY_ID)]
            gammaFractions = [[[[1.0, 0.5, 0.5] for n in xrange(MAX_DOCS_PER_QUERY + 1)] for m in xrange(MAX_DOCS_PER_QUERY + 1)] for i in xrange(MAX_DOCS_PER_QUERY)]
            
            #E-step
            for s in sessions:
                query = s.query
                layout = [False] * len(s.layout) if self.ignoreLayout else s.layout
                p_I__C_G = {False: 1.0, True: 0}
                self.queryIntentsWeights[query].append(p_I__C_G[True])
                click_seq_list = self.generate_click_seq(s.clicks, s.click_times)
                p_click = MAX_DOCS_PER_QUERY #default C_0
                for l in range(0, len(click_seq_list) + 1):
                    if l == len(click_seq_list):
                        n_click = MAX_DOCS_PER_QUERY
                    else:
                        n_click = click_seq_list[l][0]
                    m = p_click
                    n = n_click
                    i_begin = min(m,n) + 1
                    i_end = max(m,n) + 1
                    if m == MAX_DOCS_PER_QUERY: #first
                        i_begin = 0
                        i_end = n + 1
                    if n == MAX_DOCS_PER_QUERY:
                        i_end = MAX_DOCS_PER_QUERY
                    for i in range(i_begin, i_end):
                        url = s.urls[i]
                        a = self.alpha[query][url]
                        g = self.gamma[i][m][n]
                        if i == n:
                            alphaFractions[query][url][2] += 1.0
                            gammaFractions[i][m][n][2] += 1.0
                        else:
                            alphaFractions[query][url][0] += (1.0 - a) / (1.0 - a * g)
                            alphaFractions[query][url][1] += a * (1.0 - g) / (1.0 - a * g)
                            gammaFractions[i][m][n][0] += (1.0 - g) / (1.0 - a * g)
                            gammaFractions[i][m][n][1] += g * (1.0 - a) / (1.0 - a * g)
                    p_click = n_click
            
            # M-step
            sum_square_displacement = 0.0
            num_points = 0
            for q in xrange(MAX_QUERY_ID):
                for url, aF in alphaFractions[q].iteritems():
                    new_alpha = (aF[1] + aF[2]) / (aF[0] + aF[1] + aF[2])
                    sum_square_displacement += (self.alpha[q][url] - new_alpha) ** 2
                    num_points += 1
                    self.alpha[q][url] = new_alpha
            for i in xrange(MAX_DOCS_PER_QUERY):
                for m in xrange(MAX_DOCS_PER_QUERY + 1):
                    for n in xrange(MAX_DOCS_PER_QUERY):
                        gF = gammaFractions[i][m][n]
                        new_gamma = (gF[1] + gF[2]) / (gF[0] + gF[1] + gF[2])
                        sum_square_displacement += (self.gamma[i][m][n] - new_gamma) ** 2
                        num_points += 1
                        self.gamma[i][m][n] = new_gamma
            if not PRETTY_LOG:
                sys.stderr.write('M\n')
            rmsd = math.sqrt(sum_square_displacement / (num_points if TRAIN_FOR_METRIC else 1.0))
            if PRETTY_LOG:
                sys.stderr.write('Iteration: %d, RMSD: %.10f' % (iteration_count + 1, rmsd))
            else:
                print >>sys.stderr, 'Iteration: %d, RMSD: %.10f' % (iteration_count + 1, rmsd)
            del alphaFractions
            del gammaFractions
        if PRETTY_LOG:
            sys.stderr.write('\n')
        
    def _getClickProbs(self, s, possibleIntents):
        """
            Returns clickProbs list
            clickProbs[i][k] = P(C_1, ..., C_k | I=i)
        """
        clickProbs = dict((poi, []) for poi in possibleIntents)
        layout = [False] * len(s.layout) if self.ignoreLayout else s.layout
        query = s.query
        P0T = [1.0 for k in xrange(MAX_DOCS_PER_QUERY)]
        click_seq_list = self.generate_click_seq(s.clicks, s.click_times)
        p_click = MAX_DOCS_PER_QUERY
        for l in range(0, len(click_seq_list) + 1):
            if l == len(click_seq_list):
                n_click = MAX_DOCS_PER_QUERY
            else:
                n_click = click_seq_list[l][0]
            m = p_click
            n = n_click
            i_begin = min(m,n) + 1
            i_end = max(m,n) + 1
            if m == MAX_DOCS_PER_QUERY: #first
                i_begin = 0
                i_end = n + 1
            if n == MAX_DOCS_PER_QUERY:
                i_end = MAX_DOCS_PER_QUERY
            for i in range(i_begin, i_end):
                url = s.urls[i]
                a = self.alpha[query][url]
                g = self.gamma[i][m][n]
                P0T[i] *= (1.0 - a * g)
        for rank, c in enumerate(s.clicks):
            url = s.urls[rank]
            prob = {False: 0.0, True: 0.0}
            for poi in possibleIntents:
                prevProb = 1 if rank == 0 else clickProbs[poi][-1]
                if c == 0:
                    clickProbs[poi].append(prevProb * P0T[rank])
                else:
                    clickProbs[poi].append(prevProb * (1.0 - P0T[rank]))
        return clickProbs
    
    
class THCMModel(ClickModel):
    def __init__(self, ignoreIntents=True, ignoreLayout=True, explorationBias=False, step_size=0.00001):
        self.explorationBias = explorationBias
        ClickModel.__init__(self, ignoreIntents, ignoreLayout)
        self.step_size = step_size
    
    def get_model_info(self):
        ret = ""
        ret += "alpha:\t" + str(self.alpha) + "\n"
        ret += "gamma:\t" + str(self.gamma) + "\n"
        return ret
        
    def get_relevance_list(self):
        ret = []
        for q in xrange(MAX_QUERY_ID):
            for url in self.rel[q].keys():
                ret.append([q, url, self.rel[q][url]])
        return ret
    
    def generate_click_seq(self, click_list, click_time_list):
        click_seq_list = []
        for i in range(0, len(click_list)):
            if click_list[i] == 1:
                click_seq_list.append([i, click_time_list[i]])
        click_seq_list = sorted(click_seq_list, key = lambda x: x[1], reverse=False)
        return click_seq_list
    
    def train(self, sessions):
        self.rel = [defaultdict(lambda: DEFAULT_REL / 2.0) for q in xrange(MAX_QUERY_ID)]
        self.alpha = 0.5
        self.gamma = 0.5
        
        
        if not PRETTY_LOG:
            print >>sys.stderr, '-' * 80
            print >>sys.stderr, 'Start. Current time is', datetime.now()
        #use SGD as optimization algorithm
        url_gradient = [0.0 for i in xrange(MAX_DOCS_PER_QUERY)]
        for iteration_count in xrange(MAX_ITERATION_THCM):
            rel_tmp = [defaultdict(lambda: [0.0, 0.0, 0.0]) for q in xrange(MAX_QUERY_ID)]
            alpha_tmp = 0.0
            gamma_tmp = 0.0 
            #batch
            for s in sessions:
                for i in xrange(MAX_DOCS_PER_QUERY):
                    url_gradient[i] = 0.0
                alpha_gradient = 0.0
                gamma_gradient = 0.0
                query = s.query
                click_seq_list = self.generate_click_seq(s.clicks, s.click_times)
                for x in xrange(MAX_DOCS_PER_QUERY):
                    if rel_tmp[query][s.urls[x]][2] == 0:
                        rel_tmp[query][s.urls[x]][2] = 1.0
                        rel_tmp[query][s.urls[x]][1] = 0.1 * (MAX_DOCS_PER_QUERY - x) / MAX_DOCS_PER_QUERY
                if len(click_seq_list) == 0:
                    continue
                elif len(click_seq_list) == 1:#case1
                    m = click_seq_list[0][0]
                    url_m = s.urls[m]
                    url_gradient[m] += ((1.0 / self.rel[query][url_m]) if self.rel[query][url_m] > 0 else 0)
                    alpha_gradient -= (((1.0 + m) / self.alpha) if self.alpha > 0 else 0)
                    for x in xrange(MAX_DOCS_PER_QUERY):
                        rel_tmp[query][s.urls[x]][2] += (self.alpha ** (x + 1))
                    rel_tmp[query][s.urls[m]][1] += 1.0
                else:
                    m = click_seq_list[0][0]
                    j = click_seq_list[1][0]
                    url_m = s.urls[m]
                    url_gradient[m] += ((1.0 / self.rel[query][url_m]) if self.rel[query][url_m] > 0 else 0)
                    url_gradient[m] -= ((1.0 / (1.0 - self.rel[query][url_m])) if (1.0 - self.rel[query][url_m]) > 0 else 0)
                    if j > m:#case 2
                        denominator = self.alpha * (1.0 - self.alpha ** (MAX_DOCS_PER_QUERY - m - 1))
                        if abs(denominator) > 0:
                            alpha_gradient += (((m + 2.0) - (MAX_DOCS_PER_QUERY + 1.0) * (self.alpha ** (MAX_DOCS_PER_QUERY - m - 1))) / denominator)
                        alpha_gradient += ((1.0 / (1.0 - self.alpha)) if (1.0 - self.alpha) > 0 else 0)
                    else:#case 3
                        denominator = self.gamma - self.gamma ** (m + 1.0)
                        if abs(denominator) > 0:
                            gamma_gradient += ((1.0 - (m + 1.0) * (self.gamma ** m)) / denominator)
                        gamma_gradient += ((1.0 / (1.0 - self.gamma)) if (1.0 - self.gamma) > 0 else 0)
                    for t in range(1, len(click_seq_list)):
                        m = click_seq_list[t][0]
                        j = click_seq_list[t - 1][0]
                        url_m = s.urls[m]
                        url_gradient[m] += ((1.0 / self.rel[query][url_m]) if self.rel[query][url_m] > 0 else 0)
                        if t == 1:
                            for x in xrange(0, j):
                                rel_tmp[query][s.urls[x]][2] += (self.alpha ** (x + 1))
                        if t == len(click_seq_list) - 1:
                            for x in xrange(m + 1, MAX_DOCS_PER_QUERY):
                                rel_tmp[query][s.urls[x]][2] += (self.alpha ** (x - m + 1))
                        if j < m:#case 4
                            denominator = self.alpha - self.alpha ** (m + 1.0)
                            if abs(denominator) > 0:
                                alpha_gradient += ((1.0 - (m + 1.0) * (self.alpha ** m)) / denominator)
                            alpha_gradient += ((1.0 / (1.0 - self.alpha)) if (1.0 - self.alpha) > 0 else 0)
                            for x in xrange(j + 1, m):
                                rel_tmp[query][s.urls[x]][2] += (self.alpha ** (x + 1))
                        else:#case 5
                            denominator = self.gamma - self.gamma ** (MAX_DOCS_PER_QUERY - m)
                            if abs(denominator) > 0:
                                gamma_gradient += ((1.0 - (MAX_DOCS_PER_QUERY - m) * (self.gamma ** (MAX_DOCS_PER_QUERY - m - 1.0))) / denominator)
                            gamma_gradient += ((1.0 / (1.0 - self.gamma)) if (1.0 - self.gamma) > 0 else 0)
                            for x in xrange(m, j):
                                rel_tmp[query][s.urls[x]][2] += (self.gamma ** (j - x))
                        if t == 1:
                            rel_tmp[query][s.urls[m]][1] += 1.0
                            rel_tmp[query][s.urls[j]][1] += 1.0
                        else:
                            rel_tmp[query][s.urls[m]][1] += 1.0
                #update batch 
                for i in xrange(MAX_DOCS_PER_QUERY):
                    url = s.urls[i]
                    rel_tmp[query][url][0] += url_gradient[i]
                alpha_tmp += alpha_gradient
                gamma_tmp += gamma_gradient
            #update parameter
            a_tmp = self.alpha + self.step_size * alpha_tmp
            b_tmp = self.gamma + self.step_size * gamma_tmp
            if a_tmp > 0.0 and a_tmp < 1.0:
                self.alpha = a_tmp
            if b_tmp > 0.0 and b_tmp < 1.0:
                self.gamma = b_tmp
            #self.alpha += self.step_size * alpha_tmp
            #self.gamma += self.step_size * gamma_tmp
            #print "update alpha " + str(self.alpha)
            #print "update gamma " + str(self.gamma)
            for q in xrange(MAX_QUERY_ID):
                for u in self.rel[q].keys():
                    if rel_tmp[q].has_key(u):
                        self.rel[q][u] = ((rel_tmp[q][u][1] / rel_tmp[q][u][2]) if rel_tmp[q][u][2] >= rel_tmp[q][u][1] and rel_tmp[q][u][2] > 0 else self.rel[q][u] + self.step_size * rel_tmp[q][u][0])
                        self.rel[q][u] = max(min(self.rel[q][u], 0.999), 0.001)
            if PRETTY_LOG:
                sys.stderr.write('%d..' % (iteration_count + 1))
            else:
                print >>sys.stderr, 'Iteration: %d' % (iteration_count + 1)
            del rel_tmp    
            
    def _getClickProbs(self, s, possibleIntents):
        """
            Returns clickProbs list
            clickProbs[i][k] = P(C_1, ..., C_k | I=i)
        """
        clickProbs = dict((poi, []) for poi in possibleIntents)
        layout = [False] * len(s.layout) if self.ignoreLayout else s.layout
        query = s.query
        P0T = [1.0 for i in xrange(MAX_DOCS_PER_QUERY)]

        click_seq_list = self.generate_click_seq(s.clicks, s.click_times)
        if len(click_seq_list) == 0:
            tmp = self.alpha
            for i in xrange(MAX_DOCS_PER_QUERY):
                #print "no----" + str(i) + "\t" + str(tmp) + "\t" + str(self.rel[query][s.urls[i]])
                P0T[i] *= (1.0 - tmp * (self.rel[query][s.urls[i]]))
                tmp *= self.alpha
        else:
            #first click part
            c_click = click_seq_list[0][0]
            tmp = self.alpha
            for i in range(0, c_click + 1):
                P0T[i] *= (1.0 - tmp * (self.rel[query][s.urls[i]]))
                tmp *= self.alpha
            #adjacent click part
            for t in range(1, len(click_seq_list)):
                p_click = c_click
                c_click = click_seq_list[t][0]
                if p_click < c_click:
                    tmp = self.alpha
                    for i in range(p_click + 1, c_click + 1):
                        P0T[i] *= (1.0 - tmp * (self.rel[query][s.urls[i]]))
                        tmp *= self.alpha
                else:
                    tmp = self.gamma
                    for i in range(1, p_click - c_click + 1):
                        inverse_i = p_click - i
                        P0T[inverse_i] *= (1.0 - tmp * (self.rel[query][s.urls[inverse_i]]))
                        tmp *= self.gamma
            #last click part
            tmp = self.alpha
            for i in range(c_click + 1, MAX_DOCS_PER_QUERY):
                P0T[i] *= (1.0 - tmp * (self.rel[query][s.urls[i]]))
                tmp *= self.alpha
        for i in range(0, MAX_DOCS_PER_QUERY):
            P0T[i] = max(min(P0T[i], 0.999), 0.0001)
        #print str(P0T)
        for rank, c in enumerate(s.clicks):
            url = s.urls[rank]
            prob = {False: 0.0, True: 0.0}
            for poi in possibleIntents:
                prevProb = 1 if rank == 0 else clickProbs[poi][-1]
                if c == 0:
                    clickProbs[poi].append(prevProb * P0T[rank])
                else:
                    clickProbs[poi].append(prevProb * (1.0 - P0T[rank]))
        return clickProbs


class TCMAdModel(ClickModel):
    
    #gammaTypesNum = 1
    def __init__(self, B=100, ignoreIntents=True, ignoreLayout=True, explorationBias=False):
        self.explorationBias = explorationBias
        ClickModel.__init__(self, ignoreIntents, ignoreLayout)
        self.B = B
        self.NTYPE = 5
    
    def get_model_info(self):
        ret = "++++alpha\tbeta\tgamma\tlog_likelihood:\n"
        for i in xrange(MAX_DOCS_PER_QUERY / 2):
            ret += str(self.alpha[i]) + "\t" + str(self.gamma[i]) + "\t" + str(self.gamma[i]) + "\t" + str(self.log_likelihood[i]) + "\n"
        return ret
        
    def get_relevance_list(self):
        ret = []
        for q in xrange(MAX_QUERY_ID):
            for url in self.Nifc1c2m[q].keys():
                rel = self.rel_expectation(self.Nifc1c2m[q][url])
                ret.append([q, url, rel])
        return ret
    
    def sum_time_list(self, multi_list, x):
        inner_sum = 0.0
        for k in range(0, len(multi_list)):
            inner_sum += multi_list[k] * (x ** k)
        return inner_sum
    
    def rel_expectation(self, N_matrix):
        b_sum = 0.0
        b_norm = 0.0
        for b in range(1, self.B + 1):
            x = (b - 0.5) / self.B
            sum = 0.0
            for i in xrange(MAX_DOCS_PER_QUERY / 2):
                for f in xrange(2):
                    for c1 in xrange(2):
                        for c2 in xrange(2):
                            for m in xrange(2):
                                n = N_matrix[i][f][c1][c2][m]
                                if n > 0:
                                    multi_list = self.Rm_function_list[i][f][c1][c2][m]
                                    inner_sum = abs(self.sum_time_list(multi_list, x))
                                    sum += n * self.phi_log(inner_sum)
            b_sum += x * self.safe_exp(sum)
            b_norm += self.safe_exp(sum)
        rel_exp = (b_sum / b_norm) if b_norm > 0 else 0.0
        return rel_exp
            
    
    def get_ctype(self, ct1, ct2):
        ctype = 0
        c1 = 0
        c2 = 0
        f = 0
        if ct1 < 0 and ct2 < 0:
            ctype = 0
        elif ct1 >= 0 and ct2 < 0:
            ctype = 1
            c1 = 1
        elif ct1 < 0 and ct2 >= 0:
            ctype = 3
            c2 = 1
        elif ct1 <= ct2:
            ctype = 2
            c1 = 1
            c2 = 1
        else:
            ctype = 4
            c1 = 1
            c2 = 1
            f = 1
        return (ctype, c1, c2, f)
        
    def simple_find_max(self, gamma, N1, N2, N3, N4, N5):
        v_min = -1.0
        v_max = 1.0
        step_num = 200
        step_size = (v_max - v_min) / step_num
        alpha_ret = 0.0
        beta_ret = 0.0
        max_log_theta = -1e15
        for i in range(0, step_num + 1):
            for j in range(step_num / 2, step_num + 1):
                alpha_tmp = v_min + i * step_size
                beta_tmp = v_min + j * step_size
                if alpha_tmp <= beta_tmp and alpha_tmp + beta_tmp >= 0 and alpha_tmp + beta_tmp <= 1:
                    log_tmp = self.compute_likelihood_theta(alpha_tmp, beta_tmp, gamma, N1, N2, N3, N4, N5)
                    if log_tmp > max_log_theta:
                        max_log_theta = log_tmp
                        alpha_ret = alpha_tmp
                        beta_ret = beta_tmp
        return (alpha_ret, beta_ret, max_log_theta)
    
    def compute_likelihood_theta(self, alpha, beta, gamma, N1, N2, N3, N4, N5):
        ret = 0.0
        a1 = 1.0 - (7.0 / 12.0 * gamma)
        if a1 <= 0:
            return -1e15
        ret += N1 * math.log(a1)
        a2 = gamma * ((11.0 / 24.0) - (13.0 / 60.0 * beta) + (alpha / 72.0))
        if a2 <= 0:
            return -1e15
        ret += N2 * math.log(a2)
        a3 = gamma * ((13.0 * beta / 60.0) - (alpha / 72.0))
        if a3 <= 0:
            return -1e15
        ret += N3 * math.log(a3)
        a4 = gamma * ((1.0 / 8.0) - (beta / 30.0) + (alpha / 72.0))
        if a4 <= 0:
            return -1e15
        ret += N4 * math.log(a4)
        a5 = gamma * ((beta / 30.0) - (alpha / 72.0))
        if a5 <= 0:
            return -1e15
        ret += N5 * math.log(a5)
        return ret
    
    def phi_log(self, x):
        x = abs(x)
        if x == 0:
            return 0.0
        else:
            return math.log(x)
    
    def safe_exp(self, x):
        limit = 20
        if x > limit:
            x = limit
        return math.exp(x)
    
    def fill_Rm_function_list(self):
        for i in xrange(MAX_DOCS_PER_QUERY / 2):
            alpha = self.alpha[i]
            beta = self.beta[i]
            gamma = self.gamma[i]
            self.Rm_function_list[i][0][0][0][0] = [1.0 - 1.0 / 3.0 * gamma, 0.0, (-1.0 * gamma), (1.0 * gamma / 3.0)]
            self.Rm_function_list[i][0][0][0][1] = [1.0 - 1.0 / 2.0 * gamma, 0.0, 0.0, (-1.0 * gamma / 3.0)]
            self.Rm_function_list[i][0][1][0][0] = [0.0, (0.5 - 1.0 * beta / 6.0 - alpha / 12.0) * gamma, (1.0 - 0.5 * beta - alpha / 6.0) * gamma, (-0.5 + alpha / 2.0) * gamma, beta / 6.0, -1.0 * alpha * gamma / 12.0]
            self.Rm_function_list[i][0][1][0][1] = [gamma / 2.0, -1.0 * gamma *(beta / 2.0 - alpha / 3.0), 0.5 * alpha * gamma, -1.0 * gamma / 6.0, beta * gamma /6.0, alpha * gamma / 12.0]
            self.Rm_function_list[i][0][1][1][0] = [0.0, gamma * (beta / 6.0 + alpha / 12.0), gamma * (beta / 2.0 + alpha / 6.0), -0.5 * alpha * gamma, -1.0 * beta * gamma / 6.0, alpha * gamma / 12.0]
            self.Rm_function_list[i][0][1][1][1] = [0.0, gamma * (0.5 * beta - alpha / 3.0), 0.5 * alpha * gamma, 0.0, -1.0 * beta * gamma / 6.0,-1.0 * alpha * gamma / 12.0]
            self.Rm_function_list[i][0][0][1][0] = [gamma / 3.0, gamma * (-0.5 - beta / 3.0 + alpha / 4.0), gamma * (0.5 * beta - 2.0 * alpha / 3.0), gamma * (1.0 / 6.0 + 0.5 * alpha), -1.0 * beta * gamma / 6.0, -1.0 * alpha * gamma / 12.0]
            self.Rm_function_list[i][0][0][1][1] = [0.0, 0.0, 0.0, 0.5 * gamma, -1.0 * beta * gamma / 6.0, 1.0 * alpha * gamma / 12.0]
            self.Rm_function_list[i][1][1][1][0] = [0.0, gamma * (beta / 3.0 - alpha / 4.0), gamma * (2.0 * alpha / 3.0 - 0.5 * beta), -0.5 * alpha * gamma, 1.0 * beta * gamma / 6.0, 1.0 * alpha * gamma / 12.0]
            self.Rm_function_list[i][1][1][1][1] = [0.0, 0.0, 0.0, 0.0, 1.0 * beta * gamma / 6.0, -1.0 * alpha * gamma / 12.0]
            
    def fill_Um_function_list(self):
        for i in xrange(MAX_DOCS_PER_QUERY / 2):
            alpha = self.alpha[i]
            beta = self.beta[i]
            gamma = self.gamma[i]
            self.Um_function_list[i][0][0][0] = [1.0 - 2.0 * gamma / 3.0, 0.0, 0.5 * gamma, -1.0 * gamma / 3.0]
            self.Um_function_list[i][0][1][0] = [gamma * (1.0 / 3.0 - beta / 8.0 + alpha / 30.0), gamma * (0.5 - beta / 3.0), gamma * (-0.5 + beta / 4.0 - alpha / 6.0), gamma * (1.0 / 6.0 + alpha / 6.0), -1.0 * beta * gamma / 24.0, -1.0 * alpha * gamma / 30.0]
            self.Um_function_list[i][0][1][1] = [gamma * (beta / 8.0 - alpha / 30.0), beta * gamma / 3.0, gamma * (alpha / 6.0 - beta / 4.0), -1.0 * alpha * gamma / 6.0, beta * gamma / 24.0, alpha * gamma / 30.0]
            self.Um_function_list[i][0][0][1] = [gamma * (1.0 / 3.0 - beta / 8.0 + alpha / 10.0), (beta / 3.0 - 0.5), -1.0 * (beta / 4.0 + alpha / 6.0) * gamma, gamma * (1.0 / 6.0 + alpha / 6.0), beta * gamma / 24.0, -1.0 * alpha * gamma / 30.0]
            self.Um_function_list[i][1][1][1] = [beta / 8.0 - alpha / 10.0, -1.0 * beta * gamma / 3.0, gamma * (beta / 4.0 + alpha / 6.0), -1.0 * alpha * gamma / 6.0, 1.0 * beta * gamma / 24.0, alpha * gamma / 30.0]
            
            
    def train(self, sessions):
        self.alpha = [0.0 for i in xrange(MAX_DOCS_PER_QUERY / 2)]
        self.beta = [0.0 for i in xrange(MAX_DOCS_PER_QUERY / 2)]
        self.gamma = [0.0 for i in xrange(MAX_DOCS_PER_QUERY / 2)]
        self.log_likelihood = [0.0 for i in xrange(MAX_DOCS_PER_QUERY / 2)]
        
        self.Rm_function_list = [[[[[[] for m in xrange(2)] for c2 in xrange(2)] for c1 in xrange(2)] for f in xrange(2)] for i in xrange(MAX_DOCS_PER_QUERY / 2)]
        self.Um_function_list = [[[[[] for c2 in xrange(2)] for c1 in xrange(2)] for f in xrange(2)] for i in xrange(MAX_DOCS_PER_QUERY / 2)]
        
        self.Nifc1c2m = [defaultdict(lambda: [[[[[0 for m in xrange(2)] for c2 in xrange(2)] for c1 in xrange(2)] for f in xrange(2)] for i in xrange(MAX_DOCS_PER_QUERY / 2)]) for q in xrange(MAX_QUERY_ID)]
        
        if not PRETTY_LOG:
            print >>sys.stderr, '-' * 80
            print >>sys.stderr, 'Start. Current time is', datetime.now()
        
        #estimate global parameter
        N_count_map = [[0 for k in xrange(self.NTYPE)] for i in xrange(MAX_DOCS_PER_QUERY / 2)]
        for s in sessions:
            query = s.query
            for i in xrange(MAX_DOCS_PER_QUERY / 2):
                ct1 = s.click_times[i * 2]
                ct2 = s.click_times[i * 2 + 1]
                (ctype, c1, c2, f) = self.get_ctype(ct1, ct2)
                N_count_map[i][ctype] += 1
                u1 = s.urls[i * 2]
                u2 = s.urls[i * 2 + 1]
                self.Nifc1c2m[query][u1][i][f][c1][c2][0] += 1
                self.Nifc1c2m[query][u2][i][f][c1][c2][1] += 1
        #gamma
        for i in xrange(MAX_DOCS_PER_QUERY / 2):
            self.gamma[i] = (12.0 / 7.0) * (N_count_map[i][1] + N_count_map[i][2] + N_count_map[i][3] + N_count_map[i][4]) / ((N_count_map[i][0] + N_count_map[i][1] + N_count_map[i][2] + N_count_map[i][3] + N_count_map[i][4]))
        for i in xrange(MAX_DOCS_PER_QUERY / 2):
            (alpha, beta, likeli) = self.simple_find_max(self.gamma[i], N_count_map[i][0], N_count_map[i][1], N_count_map[i][2], N_count_map[i][3], N_count_map[i][4])
            self.alpha[i] = alpha
            self.beta[i] = beta
            self.log_likelihood[i] = likeli
        #**************************************
        
        #fill function coefficient list
        self.fill_Rm_function_list()
        self.fill_Um_function_list()
        
        #print self.get_model_info()
        
        if PRETTY_LOG:
            sys.stderr.write('\n')
    
    #revised version, perform better than original version
    def compute_PC2(self,q, u1, u2, i):
        prob_c = [0.0, 0.0, 0.0, 0.0] #<a1,a2> <a2,a1> <a1> <a2>
        base_norm_c = [0.0, 0.0, 0.0, 0.0]
        base_norm_u = 0.0
        ctr_sum = [0.0, 0.0]
        Ua1 = [0.0 for k in xrange(self.B)]
        Cua1 = [0.0 for k in xrange(self.B)]
        for b in range(1, self.B + 1):
            x = (b - 0.5) / self.B
            sum = [0.0, 0.0, 0.0]
            for f in xrange(2):
                for c1 in xrange(2):
                    for c2 in xrange(2):
                        n1 = self.Nifc1c2m[q][u1][i][f][c1][c2][0]
                        um_list = self.Um_function_list[i][f][c1][c2]
                        um_sum = self.sum_time_list(um_list, x)
                        sum[2] += n1 * self.phi_log(1.0 + um_sum)
            tmp_u =   self.safe_exp(sum[2])
            Ua1[b - 1] = tmp_u
            base_norm_u += tmp_u
            tmp_0 = abs(self.sum_time_list(self.Um_function_list[i][0][1][1], x))
            tmp_1 = abs(self.sum_time_list(self.Um_function_list[i][1][1][1], x))
            tmp_2 = abs(self.sum_time_list(self.Um_function_list[i][0][1][0], x))
            tmp_3 = abs(self.sum_time_list(self.Um_function_list[i][0][0][1], x))
            base_norm_c[0] += tmp_0
            base_norm_c[1] += tmp_1
            base_norm_c[2] += tmp_2
            base_norm_c[3] += tmp_3
            prob_c[0] += tmp_0 * tmp_u
            prob_c[1] += tmp_1 * tmp_u
            prob_c[2] += tmp_2 * tmp_u
            prob_c[3] += tmp_3 * tmp_u
        prob_c[0] = (prob_c[0] / (base_norm_u * base_norm_c[0])) if base_norm_u * base_norm_c[0] > 0 else 0.0
        prob_c[1] = (prob_c[1] / (base_norm_u * base_norm_c[1])) if base_norm_u * base_norm_c[1] > 0 else 0.0
        prob_c[2] = (prob_c[2] / (base_norm_u * base_norm_c[2])) if base_norm_u * base_norm_c[2] > 0 else 0.0
        prob_c[3] = (prob_c[3] / (base_norm_u * base_norm_c[3])) if base_norm_u * base_norm_c[3] > 0 else 0.0
        ctr_sum[0] = prob_c[0] + prob_c[1] + prob_c[2]
        ctr_sum[1] = prob_c[0] + prob_c[1] + prob_c[3]
        #print "Base norm u:\t" + str(base_norm_u)
        #print "Base norm c:\t" + str(base_norm_c)
        #print "Prob c:\t" + str(prob_c)
        return (ctr_sum[0], ctr_sum[1])
    
    #original paper's method
    def compute_PC(self,q, u1, u2, i):
        base_norm = [0.0, 0.0, 0.0]# Pa1, Pa2, Fa1
        ctr_sum = [0.0, 0.0]
        Pa1 = [0.0 for k in xrange(self.B)]
        Pa2 = [0.0 for k in xrange(self.B)]
        Fa1 = [0.0 for k in xrange(self.B)]
        test_sum = [0.0, 0.0, 0.0]
        for b in range(1, self.B + 1):
            x = (b - 0.5) / self.B
            sum = [0.0, 0.0, 0.0]
            for f in xrange(2):
                for c1 in xrange(2):
                    for c2 in xrange(2):
                        for m in xrange(2):
                            n1 = self.Nifc1c2m[q][u1][i][f][c1][c2][m]
                            n2 = self.Nifc1c2m[q][u2][i][f][c1][c2][m]
                            multi_list = self.Rm_function_list[i][f][c1][c2][m]
                            inner_sum = self.sum_time_list(multi_list, x)
                            sum[0] += n1 * self.phi_log(inner_sum)
                            sum[1] += n2 * self.phi_log(inner_sum)
                            if m == 0:
                                um_list = self.Um_function_list[i][f][c1][c2]
                                um_sum = self.sum_time_list(um_list, x)
                                sum[2] += n1 * self.phi_log(um_sum)
            Pa1[b - 1] = self.safe_exp(sum[0])
            Pa2[b - 1] = self.safe_exp(sum[1])
            Fa1[b - 1] = self.safe_exp(sum[2])
            if b - 1 > 0:
                Fa1[b - 1] += Fa1[b - 2]
            base_norm[0] += self.safe_exp(sum[0])
            base_norm[1] += self.safe_exp(sum[1])
            base_norm[2] += self.safe_exp(sum[2])
        for b in range(1, self.B + 1):
            Pa1[b - 1] = (Pa1[b - 1] / base_norm[0]) if base_norm[0] > 0 and Pa1[b - 1] > 0 else (1.0 / self.B)
            Pa2[b - 1] = (Pa2[b - 1] / base_norm[1]) if base_norm[1] > 0 and Pa2[b - 1] > 0 else (1.0 / self.B) 
            Fa1[b - 1] = (Fa1[b - 1] / base_norm[2]) if base_norm[2] > 0 and Fa1[b - 1] > 0 else (1.0 * b / self.B)
            test_sum[0] += Pa1[b - 1]
            test_sum[1] += Pa2[b - 1]
            test_sum[2] += Fa1[b - 1]
        #print "Base norm:\t" + str(base_norm)
        #print "Pa1:\tsum=" + str(test_sum[0]) + "\t" + str(Pa1)
        #print "Pa2:\tsum=" + str(test_sum[1]) + "\t" + str(Pa2)
        #print "Fa1:\tsum=" + str(test_sum[2]) + "\t" + str(Fa1)
        for b1 in range(1, self.B + 1):
            for b2 in range(1, self.B + 1):
                x1 = (b1 - 0.5) / self.B
                x2 = (b2 - 0.5) / self.B
                #<a2a1> part
                p_a2a1 = self.gamma[i] * x1 * x2 * max(0, abs(self.beta[i] - self.alpha[i] * (x2 - x1))) * Pa1[b1 - 1] * Pa2[b2 - 1] * Fa1[b1 - 1]
                #<a1a2> part
                p_a1a2 = self.gamma[i] * x1 * x2 * max(0, abs(self.beta[i] - self.alpha[i] * (x1 - x2))) * Pa1[b1 - 1] * Pa2[b2 - 1] * (1.0 - Fa1[b1 - 1])
                #<a1,o> part
                p_a1o = self.gamma[i] * x1 * Pa1[b1 - 1] * Pa2[b2 - 1] * (1.0 - Fa1[b1 - 1])
                #<a2,o> part
                p_a2o = self.gamma[i] * x2 * Pa1[b1 - 1] * Pa2[b2 - 1] * Fa1[b1 - 1]
                ctr_sum[0] += p_a2a1
                ctr_sum[0] += p_a1a2
                ctr_sum[0] += p_a1o
                ctr_sum[1] += p_a2a1
                ctr_sum[1] += p_a1a2
                ctr_sum[1] += p_a2o
        ctr_sum[0] /= (self.B * self.B)
        ctr_sum[1] /= (self.B * self.B)
        return (ctr_sum[0], ctr_sum[1])

    def _getClickProbs(self, s, possibleIntents):
        query = s.query
        P1T = [0.0 for i in xrange(MAX_DOCS_PER_QUERY)]
        for i in xrange(MAX_DOCS_PER_QUERY / 2):
            u1 = s.urls[i * 2]
            u2 = s.urls[i * 2 + 1]
            (ctr1, ctr2) = self.compute_PC2(query, u1, u2, i)
            P1T[i * 2] = max(min(ctr1, 0.999), 0.0001)
            P1T[i * 2 + 1] = max(min(ctr2, 0.999), 0.0001)
        #print "++++CTR:\t" + str(P1T)
        clickProbs = dict((i, []) for i in possibleIntents)
        prevClick = -1
        layout = [False] * len(s.layout) if self.ignoreLayout else s.layout
        for rank, c in enumerate(s.clicks):
            url = s.urls[rank]
            prob = {False: 0.0, True: 0.0}
            for i in possibleIntents:
                prevProb = 1 if rank == 0 else clickProbs[i][-1]
                if c == 0:
                    clickProbs[i].append(prevProb * (1 - P1T[rank]))
                else:
                    clickProbs[i].append(prevProb * P1T[rank])
            if c != 0:
                prevClick = rank
        return clickProbs
         

def load_class_map(file_name, K):
    query_class_map = {}
    in_file = open(file_name)
    wc_query_id = 0
    while True:
        line = in_file.readline()
        if not line:
            break
        arr = line.strip().split("\t")
        query = int(arr[0])
        query_class_map[query] = []
        for i in range(0, K):
            query_class_map[query].append(float(arr[1 + i]))
        if wc_query_id < query:
            wc_query_id = query
    in_file.close()
    global MAX_QUERY_ID
    MAX_QUERY_ID = wc_query_id + 1
    return (query_class_map)
    


def output_relevance(file_name, relevance_list):
    out_file = open(file_name, "w")
    for i in range(0, len(relevance_list)):
        #query  url relevance
        out_file.write(str(relevance_list[i][0]) + "\t" + str(relevance_list[i][1]) + "\t" + str(relevance_list[i][2]) + "\n")
    out_file.close()

def output_information_to_file(file_name, information):
    out_file = open(file_name, "w")
    out_file.write(information)
    out_file.close()
    
def test_model(model_obj, model_name):
    log_info = "Train: " + str(model_name) + "\n"
    print >>sys.stderr, log_info
    print log_info
    model_obj.train(sessions)
    test_info = str(model_name) + '\n' + model_obj.test(testSessions) + "\n"
    model_info = model_obj.get_model_info()
    relevance_list = model_obj.get_relevance_list()
    print >>sys.stderr, test_info
    print test_info
    output_relevance(out_dir + "/_" + str(model_name) + ".model.relevance", relevance_list)
    output_information_to_file(out_dir + "/_" + str(model_name) + ".model", test_info + model_info)
    model_obj.output_perplexity(out_dir + "/_" + str(model_name) + ".model.perplexity")
    del relevance_list
    del model_obj
    
if __name__ == '__main__':
    root_dir = ".."
    readInput = InputReader()
    #data_dir = sys.argv[1]
    data_dir = "../sample"
    out_dir = data_dir + "/output"
    if not os.path.exists(out_dir):
        os.system("mkdir " + out_dir)
    
    (query_class_map) = load_class_map(data_dir + "/query_class", CLASS_K)

    sessions = readInput(data_dir + "/train_data", query_class_map)
    testSessions = readInput(data_dir + "/test_data", query_class_map)
    print "MAX_QUERY " + str(MAX_QUERY_ID)
    del readInput       # needed to minimize memory consumption (see gc.collect() below)

    print 'Train sessions: %d, test sessions: %d' % (len(sessions), len(testSessions))
    
    if 'RevisitModelUBM' in TEST_MODELS or 'PSCM' in TEST_MODELS:
        test_model(RevisitModelUBM(), 'PSCM')
    
    if 'TCM' in TEST_MODELS:
        test_model(TCMAdModel(), 'TCM')
    
    if 'THCM' in TEST_MODELS:
        test_model(THCMModel(), 'THCMOri')

        
    if 'WCOneUBM' in TEST_MODELS or 'UBM' in TEST_MODELS:
        test_model(WCRealUbmModel(1), 'UBM')
    
    if 'NaiveModel' in TEST_MODELS:
        test_model(NaiveModel(), 'NaiveModel')
    
    if 'WCRealUBM' in TEST_MODELS:
        test_model(WCRealUbmModel(CLASS_K), 'UBMmu')
    
    if 'POMModel' in TEST_MODELS or 'POM' in TEST_MODELS:
        test_model(POMModel(), 'POM')
        
    if 'DBN' in TEST_MODELS:
        test_model(DbnModel((0.9, 0.9, 0.9, 0.9)), 'DBN')


