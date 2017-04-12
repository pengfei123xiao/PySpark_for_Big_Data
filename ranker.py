__author__ = 'haowu'
import numpy as np


class RankingDistribution:
    def __init__(self,rxi,ry,info = None):
        """
        :param rx: Ranking dis over X in format {idx:score} ie. {1:12.3, 2:23.3}
        :param ry: Ranking dis over Y in format same as rx
        :param info (str): detail information of this ranking dis
                            ie. "Ranking score of Cluster 4"
                            default if None
        """
        self.rxi = np.matrix(rxi)
        self.ry = np.matrix(ry)
        self.info = info

        # self.rxi = np.array(rxi)
        #
        # self.ry = np.array(ry)
        # self.info = info

    def getXscore(self,idx):
        # print self.rx,"???"
        return self.rx[0,idx]

    def getYscore(self,idx):
        return self.ry[0,idx]

    def genRX(self,wxy):
        self.rx = self.ry * np.transpose(wxy)
        # print self.rx

    def getRX(self):
        return self.rx

    def getHighestX(self,limit = None):
        # list = self.rx.tolist()[0]
        sort_index = np.argsort(self.rx[0])
        if limit is None:
            return sort_index
        else:
            return sort_index[:limit]

    def getHighestY(self,limit = None):
        # list = self.ry.tolist()[0]
        sort_index = np.argsort(self.ry[0])
        if limit is None:
            return sort_index
        else:
            return sort_index[:limit]


    def __str__(self):
        return "\nRanking info of X\n"+str(self.rx)+"\n Ranking info of Y\n"+str(self.ry)+"\n"

    def __repr__(self):
        return self.__str__()

class Ranker:
    """
    The interface of the ranker
    Hint: You only need to write the ranking function to
          generated (R_x_i|X_i,R_Y|X_i)
          r_X|X_i will be handle in the network.py
    """

    def __init__(self):
        pass
    def rank(self,subNetwork,nodelist):
        pass


class SimpleRanker(Ranker):
    def __init__(self):
        Ranker.__init__(self)

    def rank(self,subNetwork,nodelist):
        """
        :param subNetwork: the sub graph you want to rank
                            To be notice that the subNetwork has the same dim as the full network.
                            So we don't lose any information here
        :param nodelist: list of nodes idx (Type A) in the subnet you want to rank
                         provided this will make the result faster since we are skiping
                         all nodes that are not in the sub Network
        :return:    an RankingDistribution Object.
        """
        [wxx,wxy,wyx,wyy]=subNetwork
        # print subNetwork
        m = len(wxx)
        n = len(wyy)
        # init rX and rY
        rX = []
        rY = []
        Aury=[]
        # for x
        base = np.sum(wxy)
        # for idx in nodelist:
        for idx in range(m):
            # note that idx is the index of type X
            sumr =np.sum(wxy[idx,:])
            # sumr it the sum of row-idx
            tmp=float(sumr)/base
            rX.append(tmp)
        # print "rX is %s : " %rX

        for i in range(n):
            # print wxy[:,i]
            sumc =np.sum(wxy[:,i])

            # sumc it the sum of column-idx
            tmp=float(sumc)/base
            rY.append(tmp)
        # print type(rX), type(wxy)
        Aury.append(np.dot(np.array(rX),wxy))
        AuRy=np.array(Aury)
        # construct the distribution object
        ret = RankingDistribution(rX,rY)
        # and return it
        return ret,AuRy


