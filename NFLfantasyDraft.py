"""
An integer program that selects players for an NFL fantasy draft
"""

# import modules:
import os
import gc
from timeit import default_timer
from time import ctime
import re
import pandas as pd
import numpy as np
import pyomo.environ as po
from pyomo.opt import SolverFactory

# load the fantasy data:
START_LOAD = default_timer()
os.chdir("C:\\Users\\Nick Morris\\Desktop")
DATA = pd.read_csv("NFLfantasy2019.csv").dropna(axis=0, subset=["Name", "Position", "Bye", "Value"]).reset_index(drop=True)
SLOTS = pd.read_csv("TeamSlots.csv")
DRAFTED = pd.read_csv("DraftedPlayers.csv")

# determine which players have already been drafted by others
if DRAFTED.loc[DRAFTED["Mine"] == False].shape[0] > 0:

    # get the names of players I can't draft
    OTHERS = np.array(DRAFTED.loc[DRAFTED["Mine"] == False]["Name"])

    # update DATA to only contain players I can draft
    DATA = DATA.loc[DATA["Name"].isin(OTHERS) == False]
    del OTHERS

# collect the model data:
NAMES = np.array(DATA["Name"])    # player names
POSITIONS = np.array(SLOTS["Position"])    # position names
WEEKS = np.arange(1, 17 + 1).astype(int)    # week numbers
VALUE = pd.DataFrame(data={"Name": NAMES, "Value": np.array(DATA["Value"])})    # player value
BYE = pd.DataFrame(data=np.array(np.meshgrid(NAMES, WEEKS)).reshape(2, int(len(NAMES) * len(WEEKS))).T, columns=["Name", "Week"])    # player bye week
PLAYS = pd.DataFrame(data=np.array(np.meshgrid(NAMES, POSITIONS)).reshape(2, int(len(NAMES) * len(POSITIONS))).T, columns=["Name", "Position"])    # player positions
STARTER = pd.DataFrame(data={"Position": POSITIONS, "Starting": np.append(np.repeat(1, 7), [0])})    # starter positions
CAPACITY = SLOTS.copy()    # team size

# determine which players have already been drafted by me
if DRAFTED.loc[DRAFTED["Mine"] == True].shape[0] > 0:

    # get the names of players I drafted
    MINE = np.array(DRAFTED.loc[DRAFTED["Mine"] == True]["Name"])

    # build a table of picked players
    PICKS = pd.DataFrame(data={"Name": NAMES, "Picked": np.array(DATA["Name"].isin(MINE) * 1)})
    del MINE
else:
    # build a table of picked players
    PICKS = pd.DataFrame(data={"Name": NAMES, "Picked": np.repeat(0, len(NAMES))})

# update BYE to identify the bye week for each player
BYE["Bye"] = 0
for i in DATA.index.values:
    BYE.at[(BYE["Name"] == DATA["Name"][i]) & (BYE["Week"] == int(DATA["Bye"][i])), "Bye"] = 1

# update PLAYS to identify the position for each player
PLAYS["Plays"] = 0
for i in DATA.index.values:
    PLAYS.at[(PLAYS["Name"] == DATA["Name"][i]) & (PLAYS["Position"] == DATA["Position"][i]), "Plays"] = 1

# update flex positions in DATA
DATA["Flex"] = DATA["Position"].isin(["RB", "WR", "TE"])

# update PLAYS to include the flex and bench positions
for i in DATA.index.values:
    if DATA["Flex"][i]:
        PLAYS.at[(PLAYS["Name"] == DATA["Name"][i]) & (PLAYS["Position"] == "FLEX"), "Plays"] = 1
    PLAYS.at[(PLAYS["Name"] == DATA["Name"][i]) & (PLAYS["Position"] == "Bench"), "Plays"] = 1

# clean out garbage in RAM
del SLOTS, DRAFTED
gc.collect()

# print progress
LOAD_TIME = default_timer() - START_LOAD
del START_LOAD
print("Data loaded in " + str(np.round(LOAD_TIME, 4)) + " seconds")

# ---- build the model's data file ----

# collect the data for the model sets
START_DATA = default_timer()
SETS = {"N": NAMES, "P": POSITIONS, "W": WEEKS}

# collect the data for the model parameters
PARAMS = {"d": PICKS, "v": VALUE, "b": BYE, "p": PLAYS, "s": STARTER, "c": CAPACITY}

# open the path to where the data will be written
INPUT = "nfl_fantasy_data_" + re.sub(" ", "", re.sub(":", "_", ctime())) + ".dat"
with open(INPUT, "w") as MYFILE:

    # write out the sets
    for i in list(SETS.keys()):

        # write out the header for set i
        MYFILE.write("{}\n".format("set " + str(i) + " :="))

        # write out the values of set i
        for j in SETS[i]:
            MYFILE.write("{}\n".format(str(j)))

        # write out the tail for set i
        MYFILE.write("{}\n".format(";"))

    # write out the parameters
    for i in list(PARAMS.keys()):

        # write out the header for set i
        MYFILE.write("{}\n".format("param " + str(i) + " :="))

        # write out the values of set i
        if isinstance(PARAMS[i], pd.DataFrame):

            # write out the rows of set i
            for j in PARAMS[i].index.values:
                row = "\t".join([str(PARAMS[i][k][j]) for k in PARAMS[i].columns])
                MYFILE.write("{}\n".format(row))
            del row
        else:
            # write out the values of set i
            for j in PARAMS[i]:
                MYFILE.write("{}\n".format(str(j)))

        # write out the tail for set i
        MYFILE.write("{}\n".format(";"))

# print progress
DATA_TIME = default_timer() - START_DATA
del START_DATA, i, j
print("Data file built in " + str(np.round(DATA_TIME, 4)) + " seconds")

# ---- set up the model ----

START_MODEL = default_timer()
MODEL = po.AbstractModel()    # creates an abstract model
MODEL.name = "NFL Fantasy Draft Model"    # gives the model a name

# ---- define sets ----

MODEL.N = po.Set()    # a set of players [string]
MODEL.P = po.Set()    # a set of positions [string]
MODEL.W = po.Set()    # a set of weeks [integer]

# ---- define parameters ----

MODEL.d = po.Param(MODEL.N)    # a player was/wasn't drafted yet [binary]
MODEL.v = po.Param(MODEL.N)    # the value of a player [float]
MODEL.b = po.Param(MODEL.N, MODEL.W)    # a player isn't/is playing a week [binary]
MODEL.p = po.Param(MODEL.N, MODEL.P)    # a player can/can't play a position [binary]
MODEL.s = po.Param(MODEL.P)    # a position is/isn't a starting position [binary]
MODEL.c = po.Param(MODEL.P)    # the number of available spots for a position [integer]

# ---- define variables ----

MODEL.x = po.Var(MODEL.N, MODEL.P, MODEL.W, domain=po.Binary)    # a player is/isn't playing a position for a week [binary]
MODEL.y = po.Var(MODEL.N, domain=po.Binary)    # a player is/isn't drafted [binary]

# ---- define objective functions ----

# set up the objective function
def Value(MODEL):
    """ The total value of the team for the season """
    return sum(sum(sum(MODEL.x[i, j, k] * MODEL.v[i] * MODEL.s[j] * (1 - MODEL.b[i, k]) for i in MODEL.N) for j in MODEL.P) for k in MODEL.W)

# build the objective function
MODEL.obj = po.Objective(rule=Value, sense=po.maximize)    # a maximization problem of the objective function

# ---- define constraints ----

# set up the constraints
def Drafting(MODEL):
    """ A certain number of players must be drafted to fill all positions """
    return sum(MODEL.y[i] for i in MODEL.N) == sum(MODEL.c[j] for j in MODEL.P)

def Picked(MODEL, i):
    """ If a player was previously picked, they have been drafted """
    return MODEL.y[i] >= MODEL.d[i]

def Playing(MODEL, i, k):
    """ If a player is drafted, they are playing one position each week """
    return sum(MODEL.x[i, j, k] for j in MODEL.P) == MODEL.y[i]

def Filling(MODEL, j, k):
    """ Each week, a position must be filled in by a player """
    return sum(MODEL.x[i, j, k] for i in MODEL.N) == MODEL.c[j]

def Allowed(MODEL, i, j, k):
    """ Each week, a player can only play a position they are allowed to play """
    return MODEL.x[i, j, k] <= MODEL.p[i, j]

# build the constraints
MODEL.Draft = po.Constraint(rule=Drafting)    # the Drafting constraint applies to all players and positions
MODEL.Picks = po.Constraint(MODEL.N, rule=Picked)    # the Picked constraint applies to each player
MODEL.Play = po.Constraint(MODEL.N, MODEL.W, rule=Playing)    # the Playing constraint applies to each player and week
MODEL.Fill = po.Constraint(MODEL.P, MODEL.W, rule=Filling)    # the Filling constraint applies to each position and week
MODEL.Allow = po.Constraint(MODEL.N, MODEL.P, MODEL.W, rule=Allowed)    # the Allowed constraint applies to each player, position, and week

# print progress
MODEL_TIME = default_timer() - START_MODEL
del START_MODEL
print("Model designed in " + str(np.round(MODEL_TIME, 4)) + " seconds")

# ---- execute the solver ----

# reset the problem instance:
START_SOLVE = default_timer()
OPT = None
INSTANCE = None
RESULTS = None

# solve the problem instance:
OPT = SolverFactory("glpk")    # set up the GLPK solver
INSTANCE = MODEL.create_instance(INPUT)    # load the data set
RESULTS = OPT.solve(INSTANCE, tee=False)    # solve the problem instance
INSTANCE.solutions.load_from(RESULTS)    # store the solution results

# print progress
SOLVE_TIME = default_timer() - START_SOLVE
del START_SOLVE
print("Players drafted in " + str(np.round(SOLVE_TIME, 4)) + " seconds")

# ---- export the results ----

# get the schedule solution: MODEL.x[i, j, k]
START_SAVE = default_timer()
SCHEDULE = pd.DataFrame(data=INSTANCE.x.keys(), columns=["Name", "Position", "Week"])
SCHEDULE["Playing"] = pd.DataFrame(data=INSTANCE.x.get_values().items()).iloc[:, 1]

# get the team solution: MODEL.y[i]
TEAM = pd.DataFrame(data=INSTANCE.y.keys(), columns=["Name"])
TEAM["Drafted"] = pd.DataFrame(data=INSTANCE.y.get_values().items()).iloc[:, 1]

# add previous draft decisions to TEAM
TEAM["Previous_Pick"] = pd.DataFrame(data=INSTANCE.d.items()).iloc[:, 1]

# add player information to TEAM
TEAM = pd.concat([TEAM, DATA[["Position", "Bye", "PTSespn", "Rank", "ADP", "Value"]]], axis=1)

# export the solutions
SCHEDULE.to_csv("nfl_fantasy_schedule_" + re.sub(" ", "", re.sub(":", "_", ctime())) + ".csv", index=False)
TEAM.to_csv("nfl_fantasy_team_" + re.sub(" ", "", re.sub(":", "_", ctime())) + ".csv", index=False)

# clean out garbage in RAM
del OPT, INSTANCE, RESULTS, TEAM, SCHEDULE, DATA
gc.collect()

# print progress
SAVE_TIME = default_timer() - START_SAVE
del START_SAVE
print("Results saved in " + str(np.round(SAVE_TIME, 4)) + " seconds")
