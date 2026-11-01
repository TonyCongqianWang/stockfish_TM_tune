// Link against the candidate's built objects (excluding main.o).
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>

#include "attacks.h"
#include "position.h"
#include "search.h"
#include "timeman.h"
#include "tmparams.h"
#include "ucioption.h"

using namespace Stockfish;

static void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

int main() {
    Attacks::init();
    Position::init();
    OptionsMap options;
    options.add("nodestime", Option(0, 0, 10000));
    options.add("Move Overhead", Option(0, 0, 5000));
    options.add("Ponder", Option(false));
    Position pos;
    StateInfo state;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 8", false, &state);
    auto budget = [&](int clock, int inc, int mtg = 0) {
        Search::LimitsType limits;
        limits.time[WHITE] = limits.time[BLACK] = clock;
        limits.inc[WHITE] = limits.inc[BLACK] = inc;
        limits.movestogo = mtg;
        TimeManagement tm;
        tm.init(limits, pos, options);
        check(tm.optimum() >= 1 && tm.maximum() >= tm.optimum(), "invalid budget ordering");
        check(tm.maximum() <= std::max(1.0, TimeParams::TMClockCap / 1000.0 * clock), "clock cap violated");
        return std::pair(tm.optimum(), tm.maximum());
    };
    // With overhead off, changing TC units does not change allocation fractions.
    for (int inc : {0, 50, 100, 400}) {
        auto a = budget(10000, inc);
        auto b = budget(100000, 10 * inc);
        check(std::abs(double(b.first) - 10 * a.first) <= 10, "nominal scale invariance failed");
        check(std::abs(double(b.second) - 10 * a.second) <= 100, "maximum scale invariance failed");
    }
    auto a = budget(10000, 100);
    budget(1, 0);
    auto b = budget(10000, 100);
    check(a == b, "allocation depends on previous clock history");
    // Check a wide low-clock/increment/cyclic grid with realistic overhead.
    std::istringstream overhead("name Move Overhead value 10");
    options.setoption(overhead);
    for (int clock : {1, 5, 10, 50, 100, 500, 1500, 30000})
        for (int inc : {0, 1, 5, 15, 300, 1200})
            for (int mtg : {0, 1, 10, 40})
                budget(clock, inc, mtg);
    // Responsiveness shares relative income, and doesn't depend on the clock cap.
    TimeParams::TMElasticitySD = 500;
    TimeParams::TMElasticityInc = 1500;
    Search::LimitsType limits;
    limits.time[WHITE] = limits.time[BLACK] = 10000;
    TimeManagement tm;
    tm.init(limits, pos, options);
    check(std::abs(tm.responsiveness() - .5) < 1e-9, "sudden-death responsiveness failed");
    limits.inc[WHITE] = limits.inc[BLACK] = 10000;
    tm.init(limits, pos, options);
    check(tm.responsiveness() > 1.4 && tm.responsiveness() <= 1.5, "renewable responsiveness failed");
    std::cout << "Time-management invariants passed\n";
}
