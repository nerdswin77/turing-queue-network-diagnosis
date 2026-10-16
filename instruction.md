# Queue Network Diagnosis

Write `/root/results/solver.py`, a reusable program that reads the monitor readings of queueing networks and works out what is wrong with each one. For every network it has to find which stations have servers out of service, which stations are slowed down, which stations have a reduced capacity, which dispatch links send their jobs to the wrong station, and which readings are grossly wrong. It also has to estimate how much the actual demand of each observation period departed from the planned level and how fast the servers actually worked compared with the stored rates.

## Inputs

`/root/data` holds the development campaign, networks N1 and N2, plus `development_key.json`, which has the true answer for them in the same format as the required result. A campaign directory is laid out like this:

- `campaign.json` lists the network IDs under the key `networks`.
- `network_<ID>.json` exists for each listed ID and has `id`, `stations`, `links`, `arrival_scale` and `meters`.
  - A station has `id`, `servers`, `service_rate`, `capacity` and `external_rate`. Stations are numbered from 0 in the order of the list `stations`, so the station at position $ i $ of the list has `id` $ i $.
  - A link has `id`, `from`, `to` and `prob`. Links are numbered from 0 in the order of the list `links`, so the link at position $ i $ has `id` $ i $. The list is sorted by `from` and then by `to`. No two links share the same `from` and `to`, and no link has `from` equal to `to`.
  - `arrival_scale` has one entry per observation period, in period order: the planned arrival level of the period relative to the stored `external_rate` values.
  - A meter has `index`, `type`, `sigma`, and either `station` or `link`. Meters are numbered from 0 in the order of the list `meters`, so the meter at position $ i $ has `index` $ i $, and that is also the column of its readings in `z`.
  - The meters come in four blocks, in this order: all `occupancy` meters, then all `offered` meters, then all `lost` meters, then all `dispatch` meters. The first three types have a `station` and are sorted by station within their block. `dispatch` meters have a `link` and are sorted by link. No station or link has two meters of the same type.
  - The `sigma` is always 0.05 for `occupancy` meters, 0.04 for `offered`, 0.01 for `lost` and 0.03 for `dispatch`, in every network.
- `readings_<ID>.npz` holds the readings in an array `z` with one row per period (in period order) and one column per meter (in the order of `meters`).

Rates are in jobs per minute, and periods are counted from 0.

Every station is a first-come first-served queue. It has `servers` identical servers, and each one serves one job at a time with an exponentially distributed service time whose rate is `service_rate` times the network's service factor, described below, times the station's slowdown factor if the station is slowed down. The station has room for at most `capacity` jobs, counting the ones in service. A job that arrives when the station already holds `capacity` jobs is lost and leaves the network.

Jobs reach a station from outside at the rate `external_rate` times the period's `arrival_scale` times the period's demand factor, and over every link whose `to` is that station. The demand factor is the same for every station of the network within a period and is described below. A link carries the fraction `prob` of the jobs that finish service at its `from` station. Jobs that finish service at a station and are not carried by any of its links leave the network.

Each period is evaluated in steady state with the reduced-load model. The offered rate of a station is its outside arrival rate plus, for each link into it, `prob` times the throughput of the link's `from` station. Each station is treated as a queue with Poisson arrivals at its offered rate, which gives its mean number of jobs, its blocking probability (the probability that it holds `capacity` jobs) and its throughput, which is the offered rate times one minus the blocking probability. The offered rates are the solution of these relations for all stations at once.

The meter types measure the following:

- `occupancy`: the mean number of jobs at the station, waiting or in service.
- `offered`: the offered rate of the station.
- `lost`: the offered rate times the blocking probability.
- `dispatch`: the rate of jobs that the link's `from` station sends over the link, which is `prob` times the throughput of the `from` station. It is counted where the jobs leave, whichever station they end up at.

Reading noise is independent and Gaussian with standard deviation `sigma`.

The numbers of servers, service rates, capacities, link destinations and planned arrival levels in `network_<ID>.json` belong to the stored model of the network. The readings come from the actual configuration, which can differ from the stored model in the ways described next. All periods of one network share the same station errors, misrouted links and service factor, and differ in their arrival scale and demand factor.

In the actual configuration:

- The demand factor of a period is a number from 0.8 to 1.25. The demand factors of the periods are independent of one another and are not recorded anywhere in the inputs.
- The service factor of a network is a single number from 0.85 to 1.15 that applies to every server of every station in every period. It is not recorded in the inputs.

- A station can have servers out of service: one or two of them, always leaving at least one in service.
- A station can be slowed down: every one of its servers works at its `service_rate` times a factor between 0.5 and 0.8.
- A station can have a reduced capacity: `capacity` minus a positive whole-number cut, which must still leave room for at least `servers` plus one jobs.
- A link can be misrouted, so the jobs it carries reach a station other than its `to`. That station is neither the link's `from` nor the `to` of any link from the same `from`.
- No station is the `from`, the `to` or the actual destination of two misrouted links.
- A station has at most one error, of one of these four kinds, where a misrouted link counts as an error of its `from` station. So a station with a misrouted link has no other error.

Each network has at least 2 and at most 5 errors, not counting gross readings; the demand and service factors are not errors. It also has at most 2 gross readings. A gross reading deviates from the value of the actual configuration by between 12 and 30 times its `sigma`; every other reading deviates from it by less than 4.5 times its `sigma`.

The program must also handle three more campaigns that are not supplied here. Each has exactly 6 networks of 20 to 28 stations, with up to 45 links and 45 meters, observed in exactly 6 periods. The command is

```text
python3 -B solver.py --input-dir CAMPAIGN_DIR --output-dir OUTPUT_DIR
```

These campaigns use the same file layout, meter types and conventions as the development campaign, but with different networks, so sizes, routings, parameters, meter placements, arrival scales, demand and service factors, numbers of errors and kinds of errors all differ. `solver.py` must be self-contained and work without any other submitted file. It must use only the Python 3.11 standard library, numpy 2.4.4 and scipy 1.16.3, and it must not import any other package; an import of any other package fails. The program runs as an unprivileged user, with `HOME` and `TMPDIR` set to `OUTPUT_DIR`.

The program may read campaign data only through `CAMPAIGN_DIR` and must not modify it. Every file it writes must go to `OUTPUT_DIR`, which exists, is empty and is writable. The one exception is `/dev/shm`: the program may create entries there that are not directories and whose names start with `sem.mp-` or `psm_`, followed only by letters, digits or underscores (the names Python's multiprocessing gives its semaphores and shared-memory segments), and the temporary names of exactly `sem.` followed by six letters or digits, under which the C library creates a semaphore for an instant before renaming it. Every file the program leaves in `/dev/shm` is removed before and after every invocation, so no state can be carried from one invocation to the next.

The program has no network access. Invocations are deterministic: hash randomisation is off, the global `random` and `numpy.random` generators are seeded with 0, and numerical libraries use one thread. Each invocation must finish within 900 seconds of wall-clock time, measured from its start and not as CPU time, on 2 CPUs and 4 GB of memory, with at most 64 processes and threads, no written file larger than 16 MB, and at most 1000 entries created in `/dev/shm`.

## Required result

When the program has written the file `diagnosis.json` in `OUTPUT_DIR`, it exits with status 0. For example, a run on the development campaign with `--output-dir /root/results/development` writes the file `/root/results/development/diagnosis.json`.

The file is a JSON object with the single key `networks`. No JSON object in the file repeats a key, and no number in it is `NaN` or infinite. The value of `networks` is a JSON object with exactly one key per network ID of the campaign, and each network entry is a JSON object with exactly these keys:

- `servers_down`: an object whose keys are the decimal IDs, without leading zeros, of the stations with servers out of service, and whose values are the whole numbers of servers out of service.
- `slowdowns`: an object whose keys are the decimal IDs, without leading zeros, of the slowed stations, and whose values are their factors, numbers from 0.5 to 0.8.
- `buffer_cuts`: an object whose keys are the decimal IDs, without leading zeros, of the stations with a reduced capacity, and whose values are the whole-number cuts.
- `misroutes`: an object whose keys are the decimal IDs, without leading zeros, of the misrouted links, and whose values are the whole-number IDs of the stations their jobs actually reach.
- `demand_factors`: the list of the demand factors of the periods, in period order, one number from 0.8 to 1.25 per period.
- `service_factor`: the service factor of the network, a number from 0.85 to 1.15.
- `bad_readings`: the list of `[period, meter index]` pairs of the readings with a gross error, in strictly increasing lexicographic order.

Every value must respect the limits stated above for its kind of error or factor, and no station may carry more than one error in the result.

## Required accuracy

Every requirement below must hold for every network of the three additional campaigns; partial success is insufficient. The development campaign is there for developing and testing the program.

- `servers_down`, `buffer_cuts`, `misroutes` and `bad_readings` equal the true values exactly.
- The keys of `slowdowns` are exactly the slowed stations, and each reported factor is within 0.03 of the true factor.
- Each reported demand factor is within 0.05 of the true factor of its period, and the reported service factor is within 0.01 of the true factor.

The true values are the configuration, the demand and service factors and the gross readings that were used to generate each network's readings.

You have 18000 seconds to complete this task. Do not cheat by using online solutions or hints specific to this task.
