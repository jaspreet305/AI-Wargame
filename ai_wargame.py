from __future__ import annotations
import argparse
import copy
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from time import sleep
from typing import Tuple, TypeVar, Type, Iterable, ClassVar, Union
import random
import requests
import logging
import os

# maximum and minimum values for our heuristic scores (usually represents an end of game condition)
MAX_HEURISTIC_SCORE = 2000000000
MIN_HEURISTIC_SCORE = -2000000000

class UnitType(Enum):
    """Every unit type."""
    AI = 0
    Tech = 1
    Virus = 2
    Program = 3
    Firewall = 4

class Player(Enum):
    """The 2 players."""
    Attacker = 0
    Defender = 1

    def next(self) -> Player:
        """The next (other) player."""
        if self is Player.Attacker:
            return Player.Defender
        else:
            return Player.Attacker

class GameType(Enum):
    AttackerVsDefender = 0
    AttackerVsComp = 1
    CompVsDefender = 2
    CompVsComp = 3

##############################################################################################################

@dataclass(slots=True)
class Unit:
    player: Player = Player.Attacker
    type: UnitType = UnitType.Program
    health : int = 9
    # class variable: damage table for units (based on the unit type constants in order)
    damage_table : ClassVar[list[list[int]]] = [
        [3,3,3,3,1], # AI
        [1,1,6,1,1], # Tech
        [9,6,1,6,1], # Virus
        [3,3,3,3,1], # Program
        [1,1,1,1,1], # Firewall
    ]
    # class variable: repair table for units (based on the unit type constants in order)
    repair_table : ClassVar[list[list[int]]] = [
        [0,1,1,0,0], # AI
        [3,0,0,3,3], # Tech
        [0,0,0,0,0], # Virus
        [0,0,0,0,0], # Program
        [0,0,0,0,0], # Firewall
    ]

    def is_alive(self) -> bool:
        """Are we alive ?"""
        return self.health > 0

    def mod_health(self, health_delta : int):
        """Modify this unit's health by delta amount."""
        self.health += health_delta
        if self.health < 0:
            self.health = 0
        elif self.health > 9:
            self.health = 9

    def to_string(self) -> str:
        """Text representation of this unit."""
        p = self.player.name.lower()[0]
        t = self.type.name.upper()[0]
        return f"{p}{t}{self.health}"
    
    def __str__(self) -> str:
        """Text representation of this unit."""
        return self.to_string()
    
    def damage_amount(self, target: Unit) -> int:
        """How much can this unit damage another unit."""
        amount = self.damage_table[self.type.value][target.type.value]
        if target.health - amount < 0:
            return target.health
        return amount

    def repair_amount(self, target: Unit) -> int:
        """How much can this unit repair another unit."""
        amount = self.repair_table[self.type.value][target.type.value]
        if target.health + amount > 9:
            return 9 - target.health
        return amount

##############################################################################################################

@dataclass(slots=True)
class Coord:
    """Representation of a game cell coordinate (row, col)."""
    row : int = 0
    col : int = 0

    def col_string(self) -> str:
        """Text representation of this Coord's column."""
        coord_char = '?'
        if self.col < 16:
                coord_char = "0123456789abcdef"[self.col]
        return str(coord_char)

    def row_string(self) -> str:
        """Text representation of this Coord's row."""
        coord_char = '?'
        if self.row < 26:
                coord_char = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[self.row]
        return str(coord_char)

    def to_string(self) -> str:
        """Text representation of this Coord."""
        return self.row_string()+self.col_string()
    
    def __str__(self) -> str:
        """Text representation of this Coord."""
        return self.to_string()
    
    def clone(self) -> Coord:
        """Clone a Coord."""
        return copy.copy(self)

    def iter_range(self, dist: int) -> Iterable[Coord]:
        """Iterates over Coords inside a rectangle centered on our Coord."""
        for row in range(self.row-dist,self.row+1+dist):
            for col in range(self.col-dist,self.col+1+dist):
                yield Coord(row,col)

    def iter_adjacent(self) -> Iterable[Coord]:
        """Iterates over adjacent Coords."""
        yield Coord(self.row-1,self.col)
        yield Coord(self.row,self.col-1)
        yield Coord(self.row+1,self.col)
        yield Coord(self.row,self.col+1)

    @classmethod
    def from_string(cls, s : str) -> Coord | None:
        """Create a Coord from a string. ex: D2."""
        s = s.strip()
        for sep in " ,.:;-_":
                s = s.replace(sep, "")
        if (len(s) == 2):
            coord = Coord()
            coord.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coord.col = "0123456789abcdef".find(s[1:2].lower())
            return coord
        else:
            return None

##############################################################################################################

@dataclass(slots=True)
class CoordPair:
    """Representation of a game move or a rectangular area via 2 Coords."""
    src : Coord = field(default_factory=Coord)
    dst : Coord = field(default_factory=Coord)

    def to_string(self) -> str:
        """Text representation of a CoordPair."""
        return self.src.to_string()+" "+self.dst.to_string()
    
    def __str__(self) -> str:
        """Text representation of a CoordPair."""
        return self.to_string()

    def clone(self) -> CoordPair:
        """Clones a CoordPair."""
        return copy.copy(self)

    def iter_rectangle(self) -> Iterable[Coord]:
        """Iterates over cells of a rectangular area."""
        for row in range(self.src.row,self.dst.row+1):
            for col in range(self.src.col,self.dst.col+1):
                yield Coord(row,col)

    @classmethod
    def from_quad(cls, row0: int, col0: int, row1: int, col1: int) -> CoordPair:
        """Create a CoordPair from 4 integers."""
        return CoordPair(Coord(row0,col0),Coord(row1,col1))
    
    @classmethod
    def from_dim(cls, dim: int) -> CoordPair:
        """Create a CoordPair based on a dim-sized rectangle."""
        return CoordPair(Coord(0,0),Coord(dim-1,dim-1))
    
    @classmethod
    def from_string(cls, s : str) -> CoordPair | None:
        """Create a CoordPair from a string. ex: A3 B2"""
        s = s.strip()
        for sep in " ,.:;-_":
                s = s.replace(sep, "")
        if (len(s) == 4):
            coords = CoordPair()
            coords.src.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coords.src.col = "0123456789abcdef".find(s[1:2].lower())
            coords.dst.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[2:3].upper())
            coords.dst.col = "0123456789abcdef".find(s[3:4].lower())
            return coords
        else:
            return None

##############################################################################################################

@dataclass(slots=True)
class Options:
    """Representation of the game options."""
    dim: int = 5
    max_depth : int | None = 4
    min_depth : int | None = 2
    max_time : float | None = 5.0
    game_type : GameType = GameType.AttackerVsDefender
    alpha_beta : bool = True
    max_turns : int | None = 100
    timeout : int | None = 100
    randomize_moves : bool = True
    broker : str | None = None

##############################################################################################################

@dataclass(slots=True)
class Stats:
    """Representation of the global game statistics."""
    evaluations_per_depth : dict[int,int] = field(default_factory=dict)
    total_seconds: float = 0.0

##############################################################################################################

@dataclass(slots=True)
class Game:
    """Representation of the game state."""
    board: list[list[Unit | None]] = field(default_factory=list)
    next_player: Player = Player.Attacker
    turns_played : int = 0
    options: Options = field(default_factory=Options)
    stats: Stats = field(default_factory=Stats)
    _attacker_has_ai : bool = True
    _defender_has_ai : bool = True

    def __post_init__(self):
        """Automatically called after class init to set up the default board state."""
        is_ai = self.options.alpha_beta
        timeout = self.options.timeout
        max_turns = self.options.max_turns
        game_type = self.options.game_type

        filename = f'gameTrace-{is_ai}-{timeout}-{max_turns}.txt'
        
        if os.path.exists(filename):
            os.remove(filename)
    
        logging.basicConfig(filename=filename, level=logging.INFO, format='%(message)s')
        
        player_one = ""
        player_two = ""

        if (game_type.value == 0):
            player_one = "Human"
            player_two = "Human"
        elif (game_type.value == 1):
            player_one = "Human"
            player_two = "AI"
        elif (game_type.value == 2):
            player_one = "AI"
            player_two = "Human"
        elif (game_type.value == 3):
            player_one = "AI"
            player_two = "AI"

        table_data = [
            ["Timeout    ", f"{self.options.timeout}"],
            ["Max Turns  ", f"{self.options.max_turns}"],
            ["Alpha Beta ", f"{self.options.alpha_beta}"],
            ["Play Mode  ", f"Player 1: {player_one}, Player 2: {player_two}"],
        ]

        if (game_type.value == 1 or game_type.value == 2 or game_type.value == 3):
            table_data.append(["Heuristic", f"e0 e1 e2"])

        table_str = "\n".join(["\t".join(row) for row in table_data])

        logging.info(f"{table_str}\n")

        dim = self.options.dim
        self.board = [[None for _ in range(dim)] for _ in range(dim)]
        md = dim-1
        self.set(Coord(0,0),Unit(player=Player.Defender,type=UnitType.AI))
        self.set(Coord(1,0),Unit(player=Player.Defender,type=UnitType.Tech))
        self.set(Coord(0,1),Unit(player=Player.Defender,type=UnitType.Tech))
        self.set(Coord(2,0),Unit(player=Player.Defender,type=UnitType.Firewall))
        self.set(Coord(0,2),Unit(player=Player.Defender,type=UnitType.Firewall))
        self.set(Coord(1,1),Unit(player=Player.Defender,type=UnitType.Program))
        self.set(Coord(md,md),Unit(player=Player.Attacker,type=UnitType.AI))
        self.set(Coord(md-1,md),Unit(player=Player.Attacker,type=UnitType.Virus))
        self.set(Coord(md,md-1),Unit(player=Player.Attacker,type=UnitType.Virus))
        self.set(Coord(md-2,md),Unit(player=Player.Attacker,type=UnitType.Program))
        self.set(Coord(md,md-2),Unit(player=Player.Attacker,type=UnitType.Program))
        self.set(Coord(md-1,md-1),Unit(player=Player.Attacker,type=UnitType.Firewall))

    def clone(self) -> Game:
        """Make a new copy of a game for minimax recursion.

        Shallow copy of everything except the board (options and stats are shared).
        """
        new = copy.copy(self)
        new.board = copy.deepcopy(self.board)
        return new

    def is_empty(self, coord : Coord) -> bool:
        """Check if contents of a board cell of the game at Coord is empty (must be valid coord)."""
        return self.board[coord.row][coord.col] is None

    def get(self, coord : Coord) -> Unit | None:
        """Get contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            return self.board[coord.row][coord.col]
        else:
            return None

    def set(self, coord : Coord, unit : Unit | None):
        """Set contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            self.board[coord.row][coord.col] = unit

    def remove_dead(self, coord: Coord):
        """Remove unit at Coord if dead."""
        unit = self.get(coord)
        if unit is not None and not unit.is_alive():
            self.set(coord,None)
            if unit.type == UnitType.AI:
                if unit.player == Player.Attacker:
                    self._attacker_has_ai = False
                else:
                    self._defender_has_ai = False

    def mod_health(self, coord : Coord, health_delta : int):
        """Modify health of unit at Coord (positive or negative delta)."""
        target = self.get(coord)
        if target is not None:
            target.mod_health(health_delta)
            self.remove_dead(coord)

    def is_valid_move(self, coords : CoordPair) -> Tuple[bool, Union[str, None]]:
        """Validate a move expressed as a CoordPair. TODO: WRITE MISSING CODE!!!"""
        destination_unit = self.get(coords.dst)
        source_unit = self.get(coords.src)
        up_move = coords.dst.row == coords.src.row - 1 and coords.dst.col == coords.src.col
        down_move = coords.dst.row == coords.src.row + 1 and coords.dst.col == coords.src.col
        left_move = coords.dst.col == coords.src.col - 1 and coords.dst.row == coords.src.row
        right_move = coords.dst.col == coords.src.col + 1 and coords.dst.row == coords.src.row
        self_destruct = coords.dst.row == coords.src.row and coords.dst.col == coords.src.col
        
        if not self.is_valid_coord(coords.src) or not self.is_valid_coord(coords.dst):
            return (False, None)

        if source_unit is None or source_unit.player != self.next_player: 
            return (False, None)
        
        # Check that attackers only move up or left by one position
        if source_unit and source_unit.player == Player.Attacker:
            valid_virus_move = (source_unit.type is UnitType.Virus) and (down_move or right_move)
            if not (up_move or left_move or self_destruct or valid_virus_move):
                return (False, None)
            
        # Check that defenders only move down or right by one position
        elif source_unit and source_unit.player == Player.Defender:
            valid_tech_move = (source_unit.type is UnitType.Tech) and (left_move or up_move)
            if not (down_move or right_move or self_destruct or valid_tech_move):
                return (False, None)

        # Check if the unit is engaged in combat
        for adjacent_coord in coords.src.iter_adjacent():
            adjacent_unit = self.get(adjacent_coord)
            if adjacent_unit and adjacent_unit.player != source_unit.player:
                if source_unit.type in {UnitType.AI, UnitType.Firewall, UnitType.Program}:
                    # If a spot is empty or occupied by a friendly unit
                    if destination_unit is None or destination_unit.player == self.next_player:
                        if not self_destruct:
                            return (False, "Invalid Move: Combat mode!")
            
        if destination_unit is None:
            logging.info(f"{self.next_player.name}'s move: {coords.src} to {coords.dst}.\n")
            return (True, None)
        elif (source_unit == destination_unit) and (coords.src == coords.dst):
            logging.info(f"{self.next_player.name}'s {source_unit.type.name} self-destructing at: {coords.src}!\n")
            return (True, "self-destruct")
        elif source_unit.player == destination_unit.player:
            logging.info(f"{self.next_player.name}'s {source_unit.type.name} at {coords.src} repairing friendly {destination_unit.type.name} at {coords.dst}.\n")
            return (True, "repair")
        elif source_unit.player != destination_unit.player:
            logging.info(f"{self.next_player.name}'s {source_unit.type.name} at {coords.src} attacking enemy's {destination_unit.type.name} at {coords.dst}.\n")
            return (True, "attack")


    def perform_move(self, coords : CoordPair) -> Tuple[bool,str]:
        """Validate and perform a move expressed as a CoordPair. TODO: WRITE MISSING CODE!!!"""
        valid_move, move_type = self.is_valid_move(coords)
        if valid_move:
            if move_type == "repair":
                unit_to_repair = self.get(coords.dst)
                repairing_unit = self.get(coords.src)

                # Calculate the amount of repair done by repairing_unit to unit_to_repair
                repair_amount = repairing_unit.repair_table[repairing_unit.type.value][unit_to_repair.type.value]
                if repair_amount == 0 or unit_to_repair.health == 9: 
                    return (False, "invalid move")
                
                unit_to_repair.mod_health(repair_amount)
                logging.info(f"Repair amount: {repair_amount}")
                # Remove dead units from the board if the repair fully heals the unit
                self.remove_dead(coords.dst)
                
                return (True, "repair")
            elif move_type == "attack":
                attacking_unit = self.get(coords.src)
                unit_to_attack = self.get(coords.dst)

                # Damage done by the attack_unit to the unit_to_attack
                attack_damage = attacking_unit.damage_table[attacking_unit.type.value][unit_to_attack.type.value]
                unit_to_attack.mod_health(-attack_damage)

                # Counter attack damage
                counter_attack_damage = unit_to_attack.damage_table[unit_to_attack.type.value][attacking_unit.type.value]
                attacking_unit.mod_health(-counter_attack_damage)

                logging.info(f"Combat damage: {attack_damage}")
                logging.info(f"Counter attack damage: {counter_attack_damage}")
                # Remove dead units from the board
                self.remove_dead(coords.src)
                self.remove_dead(coords.dst)
                
                return (True,"attack")
            elif move_type == "self-destruct":
                unit_to_self_destruct = self.get(coords.src)
                unit_to_self_destruct.mod_health(-unit_to_self_destruct.health)
                self.remove_dead(coords.src)

                # Get coordinates of adjacent units
                for coord in coords.src.iter_range(1):
                    unit = self.get(coord)
                    if unit is not None:
                        unit.mod_health(-2)
                        self.remove_dead(coord)

                return (True,"self-destruct")
            else:
                self.set(coords.dst,self.get(coords.src))
                self.set(coords.src,None)
                return (True,"")
        return (False,"invalid move")

    def next_turn(self):
        """Transitions game to the next turn."""
        self.next_player = self.next_player.next()
        self.turns_played += 1

    def to_string(self) -> str:
        """Pretty text representation of the game."""
        dim = self.options.dim
        output = ""
        output += f"Next player: {self.next_player.name}\n"
        output += f"Turns played: {self.turns_played}\n"
        coord = Coord()
        output += "\n   "
        for col in range(dim):
            coord.col = col
            label = coord.col_string()
            output += f"{label:^3} "
        output += "\n"
        for row in range(dim):
            coord.row = row
            label = coord.row_string()
            output += f"{label}: "
            for col in range(dim):
                coord.col = col
                unit = self.get(coord)
                if unit is None:
                    output += " .  "
                else:
                    output += f"{str(unit):^3} "
            output += "\n"
        logging.info(output)
        return output

    def __str__(self) -> str:
        """Default string representation of a game."""
        return self.to_string()
    
    def is_valid_coord(self, coord: Coord) -> bool:
        """Check if a Coord is valid within out board dimensions."""
        dim = self.options.dim
        if coord.row < 0 or coord.row >= dim or coord.col < 0 or coord.col >= dim:
            return False
        return True

    def read_move(self) -> CoordPair:
        """Read a move from keyboard and return as a CoordPair."""
        while True:
            s = input(F'Player {self.next_player.name}, enter your move: ')
            coords = CoordPair.from_string(s)
            if coords is not None and self.is_valid_coord(coords.src) and self.is_valid_coord(coords.dst):
                return coords
            else:
                print('Invalid coordinates! Try again.')
    
    def human_turn(self):
        """Human player plays a move (or get via broker)."""
        if self.options.broker is not None:
            print("Getting next move with auto-retry from game broker...")
            while True:
                mv = self.get_move_from_broker()
                if mv is not None:
                    (success,result) = self.perform_move(mv)
                    print(f"Broker {self.next_player.name}: ",end='')
                    print(result)
                    if success:
                        self.next_turn()
                        break
                sleep(0.1)
        else:
            while True:
                mv = self.read_move()
                (success,result) = self.perform_move(mv)
                if success:
                    print(f"Player {self.next_player.name}: ",end='')
                    print(result)
                    self.next_turn()
                    break
                else:
                    print("The move is not valid! Try again.")

    def computer_turn(self) -> CoordPair | None:
        """Computer plays a move."""
        mv = self.suggest_move()
        if mv is not None:
            (success,result) = self.perform_move(mv)
            if success:
                print(f"Computer {self.next_player.name}: ",end='')
                print(result)
                self.next_turn()
        return mv

    def player_units(self, player: Player) -> Iterable[Tuple[Coord,Unit]]:
        """Iterates over all units belonging to a player."""
        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is not None and unit.player == player:
                yield (coord,unit)

    def is_finished(self) -> bool:
        """Check if the game is over."""
        return self.has_winner() is not None

    def has_winner(self) -> Player | None:
        """Check if the game is over and returns winner"""
        if self.options.max_turns is not None and self.turns_played >= self.options.max_turns:
            return Player.Defender
        if self._attacker_has_ai:
            if self._defender_has_ai:
                return None
            else:
                return Player.Attacker    
        return Player.Defender

    def move_candidates(self) -> Iterable[CoordPair]:
        """Generate valid move candidates for the next player."""
        move = CoordPair()
        for (src,_) in self.player_units(self.next_player):
            move.src = src
            for dst in src.iter_adjacent():
                move.dst = dst
                valid_move, _ = self.is_valid_move(move)
                if valid_move:
                    yield move.clone()
            move.dst = src
            yield move.clone()

    def random_move(self) -> Tuple[int, CoordPair | None, float]:
        """Returns a random move."""
        move_candidates = list(self.move_candidates())
        random.shuffle(move_candidates)
        if len(move_candidates) > 0:
            return (0, move_candidates[0], 1)
        else:
            return (0, None, 0)

    def evaluate_board(self) -> int:
        VP1, TP1, FP1, PP1, AIP1 = self.count_units(Player.Attacker)
        VP2, TP2, FP2, PP2, AIP2 = self.count_units(Player.Defender)

        e0 = (3*VP1 + 3*TP1 + 3*FP1 + 3*PP1 + 9999*AIP1) - (3*VP2 + 3*TP2 + 3*FP2 + 3*PP2 + 9999*AIP2)
        return e0

    def count_units(self, player: Player) -> Tuple[int, int, int, int, int]:
        VP, TP, FP, PP, AIP = 0, 0, 0, 0, 0

        for (_,unit) in self.player_units(player):
            if unit.type == UnitType.Virus:
                VP += 1
            elif unit.type == UnitType.Tech:
                TP += 1
            elif unit.type == UnitType.Firewall:
                FP += 1
            elif unit.type == UnitType.Program:
                PP += 1
            elif unit.type == UnitType.AI:
                AIP += 1
        return VP, TP, FP, PP, AIP
    
    def minimax(self, depth, is_maximizing):
        if depth == 0:
            return self.evaluate_board(), None  # returning a move of None since we don't need it at leaf nodes

        possible_moves = list(self.move_candidates())
        best_move = possible_moves[0]

        if is_maximizing:
            max_eval = -float('inf')
            for move in possible_moves:
                cloned_game = self.clone()
                cloned_game.perform_move(move)
                eval, _ = cloned_game.minimax(depth - 1, False)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in possible_moves:
                cloned_game = self.clone()
                cloned_game.perform_move(move)
                eval, _ = cloned_game.minimax(depth - 1, True)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
            return min_eval, best_move
    
    def suggest_move(self) -> CoordPair | None:
        """Suggest the next move using minimax alpha beta. TODO: REPLACE RANDOM_MOVE WITH PROPER GAME LOGIC!!!"""
        start_time = datetime.now()
        # (score, move, avg_depth) = self.random_move()
        (score, move) = self.minimax(3, self.next_player == Player.Attacker)
        
        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        self.stats.total_seconds += elapsed_seconds
        print(f"Heuristic score: {score}")
        # print(f"Average recursive depth: {avg_depth:0.1f}")
        print(f"Evals per depth: ",end='')
        for k in sorted(self.stats.evaluations_per_depth.keys()):
            print(f"{k}:{self.stats.evaluations_per_depth[k]} ",end='')
        print()
        total_evals = sum(self.stats.evaluations_per_depth.values())
        if self.stats.total_seconds > 0:
            print(f"Eval perf.: {total_evals/self.stats.total_seconds/1000:0.1f}k/s")
        print(f"Elapsed time: {elapsed_seconds:0.1f}s")
        return move

    def post_move_to_broker(self, move: CoordPair):
        """Send a move to the game broker."""
        if self.options.broker is None:
            return
        data = {
            "from": {"row": move.src.row, "col": move.src.col},
            "to": {"row": move.dst.row, "col": move.dst.col},
            "turn": self.turns_played
        }
        try:
            r = requests.post(self.options.broker, json=data)
            if r.status_code == 200 and r.json()['success'] and r.json()['data'] == data:
                # print(f"Sent move to broker: {move}")
                pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")

    def get_move_from_broker(self) -> CoordPair | None:
        """Get a move from the game broker."""
        if self.options.broker is None:
            return None
        headers = {'Accept': 'application/json'}
        try:
            r = requests.get(self.options.broker, headers=headers)
            if r.status_code == 200 and r.json()['success']:
                data = r.json()['data']
                if data is not None:
                    if data['turn'] == self.turns_played+1:
                        move = CoordPair(
                            Coord(data['from']['row'],data['from']['col']),
                            Coord(data['to']['row'],data['to']['col'])
                        )
                        print(f"Got move from broker: {move}")
                        return move
                    else:
                        # print("Got broker data for wrong turn.")
                        # print(f"Wanted {self.turns_played+1}, got {data['turn']}")
                        pass
                else:
                    # print("Got no data from broker")
                    pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")
        return None

##############################################################################################################

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(
        prog='ai_wargame',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max_depth', type=int, help='maximum search depth')
    parser.add_argument('--max_time', type=float, help='maximum search time')
    parser.add_argument('--game_type', type=str, default="manual", help='game type: auto|attacker|defender|manual')
    parser.add_argument('--broker', type=str, help='play via a game broker')
    parser.add_argument('--max_turns', type=int, help='maximum number of turns')
    parser.add_argument('--timeout', type=float, help='timeout between turns for ai')
    parser.add_argument('--alpha_beta', type=bool, help='is alpha-beta or minimax')
    parser.add_argument('--play_mode', type=str, help='play mode for the game')
    args = parser.parse_args()

    # For testing purposes
    game_type = GameType.AttackerVsComp

    print(f"\n\n--------- Welcome to the AI War Game! ---------\n\n")

    max_turns = int(input('Please enter the max number of turns: '))
    game_type = int(input(f'\nPlease enter the game type (1-4)\n1. Attacker vs Defender\n2. Attacker vs AI\n3. AI vs Defender\n4. AI vs AI\n'))
    alpha_beta = True
    timeout = 100

    if (game_type == 2 or game_type == 3 or game_type == 4):
        alpha_beta = int(input(f'\nPlease enter 1 for alpha-beta or 0 for minimax: '))
        if alpha_beta == 0:
            alpha_beta = False
        else:
            alpha_beta = True
        timeout = int(input(f'\nPlease enter the timeout between turns: '))

    if game_type == 1:
        game_type = GameType.AttackerVsDefender
    elif game_type == 2:
        game_type = GameType.AttackerVsComp
    elif game_type == 3:
        game_type = GameType.CompVsDefender
    elif game_type == 4:
        game_type = GameType.CompVsComp

    options = Options(max_turns=max_turns, game_type=game_type, alpha_beta=alpha_beta, timeout=timeout)

    # override class defaults via command line options
    if args.max_depth is not None:
        options.max_depth = args.max_depth
    if args.max_time is not None:
        options.max_time = args.max_time
    if args.broker is not None:
        options.broker = args.broker

    # create a new game
    game = Game(options=options)

    # the main game loop
    while True:
        print()
        print(game)
        winner = game.has_winner()
        if winner is not None:
            logging.info(f"{winner.name} wins in {game.turns_played} turns!")
            print(f"{winner.name} wins!")
            break
        if game.options.game_type == GameType.AttackerVsDefender:
            game.human_turn()
        elif game.options.game_type == GameType.AttackerVsComp and game.next_player == Player.Attacker:
            game.human_turn()
        elif game.options.game_type == GameType.CompVsDefender and game.next_player == Player.Defender:
            game.human_turn()
        else:
            player = game.next_player
            move = game.computer_turn()
            if move is not None:
                game.post_move_to_broker(move)
            else:
                print("Computer doesn't know what to do!!!")
                exit(1)

##############################################################################################################

if __name__ == '__main__':
    main()