'use strict';

Array.prototype.back = function () {
    if (this.length === 0) return undefined;
    return this[this.length - 1];
};

class Metric {
    constructor(metricJson) {
        this.key = metricJson.key;
        this.name = metricJson.name;
        this.higherIsBetter = metricJson.higher_is_better;
        this.formatter = metricJson.formatter || {};
        this.displayInSummary = metricJson.display_in_summary;
        this.displayInExample = metricJson.display_in_example;
    }

    toValue(value) { return value; }

    compare(lhs, rhs) {  // returns (lhs < rhs), where < points towards the better metric
        const l = this.toValue(lhs), r = this.toValue(rhs);
        if (this.higherIsBetter === true) return r - l;
        if (this.higherIsBetter === false) return l - r;
    }

    difference(lhs, rhs) { return lhs - rhs; }
}

class IntMetric extends Metric {
    format(value) { return value.toString(); }
}

class FloatMetric extends Metric {
    format(value) {
        if (this.formatter.fixed !== undefined) value = value.toFixed(this.formatter.fixed);
        return value.toString();
    }
}

class PortionMetric extends Metric {
    format(value) {
        return value.correct + " / " + value.total;
    }

    // toValue(value) { return value.correct / value.total; }
    toValue(value) { return value.correct; }

    difference(lhs, rhs) {
        return {
            correct: lhs.correct - rhs.correct,
            total: Math.max(lhs.total, rhs.total),
        };
    }
}

class ConfusionMatMetric extends Metric {
    format(value) {
        const tp = value.true_positive, fp = value.false_positive, fn = value.false_negative;
        return "P: " + (tp + " / " + (tp + fp)) + ", R: " + (tp + " / " + (tp + fn));
    }

    toValue(value) {  // F1
        return 2 * value.true_positive / Math.max(1, 2 * value.true_positive + value.false_positive + value.false_negative);
    }
}

let App = angular.module('App', [
    'ngRoute',
    'ngSanitize',
]).config(['$routeProvider', '$locationProvider', '$sanitizeProvider',
    function ($routeProvider, $locationProvider, $sanitizeProvider) {
        $locationProvider.hashPrefix('!');
        $sanitizeProvider.addValidAttrs(["style"]);
        $routeProvider
            .when('/summary', {
                templateUrl: "summary.html",
                activeTab: "summary",
            })
            .when('/example/:id?', {
                templateUrl: "example.html",
                activeTab: "example",
            })
            .when('/compare', {
                templateUrl: "compare.html",
                activeTab: "compare",
            })
            .otherwise({
                redirectTo: '/summary',
            });
    }
]);

App.directive('ngPrism', function () {
    return {
        restrict: 'E',
        template: `<pre><code class="language-{{language}}"></code></pre>`,
        scope: {
            language: "@",
            code: "=",
            varMap: "=?",
        },
        link: {
            post: function ($scope, $element) {
                const codeElement = $element.find("code").get(0);
                $scope.$watch("code", function () {
                    // We have to manually set code contents, because Prism removes stuff that AngularJS uses to keep
                    // track of bindings.
                    let $parent = $element.parent();
                    while ($parent.attr("name") === undefined) $parent = $parent.parent();
                    const systemName = $parent.attr("name");
                    codeElement.innerHTML = $scope.code;
                    Prism.highlightElement(codeElement);

                    if ($scope.varMap) {
                        let src = codeElement.innerHTML;
                        for (const [varId, varNames] of Object.entries($scope.varMap)) {
                            let decompSpan = "<span class='decompiled-var'>" + varNames[0] + "</span>";
                            let oracleSpan = "<span class='oracle-var'>" + varNames[1] + "</span>";
                            src = src.replace(new RegExp(varId, 'g'), decompSpan + oracleSpan);
                        }
                        codeElement.innerHTML = src;
                    }
                });
            },
        },
    };
});

App.directive('ngCheck', function () {
    return {
        restrict: 'E',
        scope: {
            pred: "=",
            label: "=",
            cond: "=",
        },
        transclude: true,
        template: `
            <span ng-style="{color: cond || pred == label ? 'green' : 'red'}">
                <code>{{pred}}</code> {{cond || pred == label ? '✓' : '✗'}} <span ng-transclude></span>
            </span>
        `,
    };
});

App.directive('ngMetricTable', function () {
    return {
        restrict: 'E',
        scope: {
            values: "=",
            metrics: "=",
            names: "=",
            excludeValue: "=?",
            align: "@?",
        },
        template: `
            <table class="ui celled definition table metric-table">
                <thead><tr>
                    <th></th>
                    <th class="center aligned" ng-repeat="x in names">{{x}}</th>
                </tr></thead>
                <tbody>
                    <tr ng-if="metrics == false">
                        <td></td>
                        <td class="center aligned" colspan="{{values.length}}">No metrics selected</td>
                    </tr>
                    <tr ng-repeat="metric in metrics track by $index" ng-init="rowIdx = $index">
                        <td>
                            <i ng-if="metric.higherIsBetter === true" class="green icon caret up"></i>
                            <i ng-if="metric.higherIsBetter === false" class="red icon caret down"></i>
                            {{metric.name}}
                        </td>
                        <td ng-repeat="value in formattedValues[rowIdx] track by $index" ng-init="colIdx = $index"
                            class="{{cellClasses[rowIdx][colIdx]}}">
                            {{value}}
                        </td>
                    </tr>
                </tbody>
            </table>
        `,
        link: function ($scope) {
            $scope.$watchGroup(["names", "metrics", "values"], (newValues, oldValues) => {
                if (newValues.some(angular.isUndefined)) return;
                if (angular.isUndefined($scope.excludeValue))
                    $scope.excludeValue = $scope.names.map(_ => false);
                $scope.formattedValues = [];
                $scope.cellClasses = [];
                for (const metric of $scope.metrics) {
                    const rowValues = $scope.values.map(dict => metric.toValue(dict[metric.key]));
                    const rowAccountedValues = rowValues.filter((_, i) => !$scope.excludeValue[i]);
                    const rowFormattedValues = $scope.values.map(dict => metric.format(dict[metric.key]));
                    let bestValue = null, worstValue = null;
                    if (metric.higherIsBetter !== null) {
                        bestValue = Math.max(...rowAccountedValues);
                        worstValue = Math.min(...rowAccountedValues);
                        if (metric.higherIsBetter === false)
                            [bestValue, worstValue] = [worstValue, bestValue];
                    }
                    let rowClasses = [];
                    for (let idx = 0; idx < rowValues.length; ++idx) {
                        const value = rowValues[idx];
                        let classes = [];
                        if (!$scope.excludeValue[idx]) {
                            if (value === bestValue) classes.push("positive");
                            else if (value === worstValue) classes.push("negative");
                        }
                        if ($scope.align) classes.push($scope.align, "aligned");
                        rowClasses.push(classes.join(" "));
                    }
                    $scope.formattedValues.push(rowFormattedValues);
                    $scope.cellClasses.push(rowClasses);
                }
            });
        },
    };
});

App.directive('ngHypothesis', function () {
    return {
        restrict: 'E',
        scope: {
            data: "=",
            target: "=",
            metrics: "=",
            varMap: "=?",
        },
        templateUrl: "hypothesis.html",
    };
});

App.controller('AccordionCtrl', ['State', '$scope', '$element', function (State, $scope, $element) {
    let _active = State.getPersistenceState("accordion-" + $element.attr("id"));
    $scope.active = {
        setDefault: function (key, value) {
            if (_active[key] === undefined)
                _active[key] = value;
        },
        get: function (key) {
            return _active[key];
        },
        toggle: function (key) {
            _active[key] = !_active[key];
            _active.$save();
        },
    };
}]).directive('ngAccordion', function () {
    return {
        controller: 'AccordionCtrl',
        restrict: 'E',
        scope: {
            exclusive: "=?",
            defaultActive: "=?",
        },
        transclude: true,
        template: `
            <div class="ui styled fluid accordion" ng-transclude>
            </div>
        `,
        link: {
            pre: function ($scope, $element) {
                if (angular.isUndefined($scope.defaultActive))
                    $scope.defaultActive = false;
                if (angular.isUndefined($scope.exclusive))
                    $scope.exclusive = true;
                $scope.$$postDigest(() => $element.accordion({
                    exclusive: $scope.exclusive,
                }));
            }
        },
    };
}).directive('ngAccordionEntry', function () {
    return {
        require: '^ngAccordion',
        restrict: 'E',
        scope: {
            id: "@",
            name: "@",
            defaultActive: "=?",
        },
        transclude: true,
        template: `
            <div class="title" ng-click="active.toggle(id)">
                <i class="dropdown icon"></i>
                {{name}}
            </div>
            <div class="content" ng-transclude></div>
        `,
        link: {
            pre: function ($scope, $element) {
                const $parentScope = $element.parent().scope();
                if (angular.isUndefined($scope.defaultActive))
                    $scope.defaultActive = $parentScope.defaultActive;
                if (angular.isUndefined($scope.exclusive))
                    $scope.exclusive = true;

                $scope.active = $parentScope.active;
                $scope.active.setDefault($scope.id, $scope.defaultActive);
                if ($scope.active.get($scope.id))
                    $element.find("div").addClass("active");
            },
        },
    };
});

App.factory('State', ['$http', '$timeout', function ($http, $timeout) {
    let state = {
        ready: false,
        metrics: null,
        systems: null,
        examples: [],
    };

    class Persistence {
        constructor(key) {
            this.$key = key;
            const loadObj = angular.fromJson(localStorage.getItem(key) || "{}");
            for (let attr in loadObj)
                if (loadObj.hasOwnProperty(attr))
                    this[attr] = loadObj[attr];
        }

        // Always remember to $save after changes!
        $save() {
            let dumpObj = {};
            for (let attr in this)
                if (this.hasOwnProperty(attr) && !attr.startsWith("$"))
                    dumpObj[attr] = this[attr];
            localStorage.setItem(this.$key, angular.toJson(dumpObj));
        }
    }

    const metricClass = {
        int: IntMetric,
        float: FloatMetric,
        portion: PortionMetric,
        confusion_mat: ConfusionMatMetric,
    };

    $http.get("static/data/eval.json.gz", {
        headers: {
            "Content-Encoding": "gzip",
        },
        responseType: "arraybuffer",
    }).then(response => {
        let data = response.data;
        if (response.headers("content-type") === "application/json") {
            // Manually parse data if the server does not support compressed HTTP response (e.g., local server).
            data = pako.inflate(response.data);
        }
        data = JSON.parse(new TextDecoder().decode(data));
        state.examples = data.examples;
        state.metrics = [];
        for (const metric of data.metrics)
            state.metrics.push(new metricClass[metric.type](metric));
        state.systems = data.systems;
        state.ready = true;
    });

    return {
        getAllExamples: () => state.examples,
        getExample: (idx) => state.examples[idx],
        getMetrics: () => state.metrics,
        getSystems: () => state.systems,
        getCount: () => state.examples.length,
        isReady: () => state.ready,
        lastIndex: 1,
        getPersistenceState: (id) => {
            return new Persistence(id);
        }
    };
}]);

App.run(['$route', '$rootScope', '$location', function ($route, $rootScope, $location) {
    // Add option in `$location.path` to prevent reload when updating path.
    const original = $location.path;
    $location.path = function (path, preventReload) {
        if (preventReload === true) {
            const lastRoute = $route.current;
            const unsubscribe = $rootScope.$on('$locationChangeSuccess', function () {
                $route.current = lastRoute;
                unsubscribe();
            });
        }
        return original.apply($location, [path]);
    };
}]);

App.controller('MainCtrl', ['State', '$route', '$scope', '$timeout', function (State, $route, $scope, $timeout) {
    $scope.$route = $route;
    $scope.links = [
        {id: "summary", name: "Summary", url: "/summary"},
        {id: "example", name: "Browse Examples", url: "/example"},
        {id: "compare", name: "Compare Systems", url: "/compare"},
    ];

    $scope.isReady = State.isReady;
    $scope.getCount = State.getCount;

    $scope.$watch(State.isReady, function (newVal, oldVal) {
        if (newVal !== oldVal && newVal) {
            $("#loading-message").transition({
                animation: "fade",
                onComplete: () => $("#load-success-message").transition({
                    animation: "fade",
                    onComplete: () => $timeout(() => $(".success.message").transition("fade"), 1500),
                }),
            });
        }
    });

    const persistence = State.getPersistenceState("_main_ctrl");
    $scope.showOracleVar = persistence.showOracleVar || false;
    $scope.updateVarName = function () {
        persistence.showOracleVar = $scope.showOracleVar;
        persistence.$save();
        if ($scope.showOracleVar) {
            $(".decompiled-var").addClass("hide");
            $(".oracle-var").removeClass("hide");
        } else {
            $(".decompiled-var").removeClass("hide");
            $(".oracle-var").addClass("hide");
        }
    };
    // Update variable names after each digest cycle, so the code is fully-loaded.
    $scope.$watch(() => $scope.$$postDigest($scope.updateVarName));
}]);

App.controller('SummaryCtrl', ['State', '$scope', function (State, $scope) {
    $scope.summaryNames = State.getSystems().map(system => system.name);
    $scope.summaryValues = State.getSystems().map(system => system.metrics);
    $scope.summaryMetrics = State.getMetrics().filter(metric => metric.displayInSummary);
}]);

App.controller('ExampleCtrl', ['State', '$location', '$route', '$routeParams', '$scope', '$timeout', function (State, $location, $route, $routeParams, $scope, $timeout) {
    $scope.exampleCount = State.getCount();
    let _firstTime = true;

    function validateIndex(idx) {
        idx = parseInt(idx);
        if (!Number.isNaN(idx) && (1 <= idx && idx <= $scope.exampleCount))
            return idx;
        return null;
    }

    $scope.metrics = State.getMetrics().filter(metric => metric.displayInExample);
    $scope.systems = State.getSystems();
    $scope.systemNames = State.getSystems().map(system => system.name);

    $scope.gotoExampleIdx = null;
    $scope.switchExample = function (idx) {
        if (idx === $scope.idx) return;
        $scope.idx = idx;
        State.lastIndex = idx;
        $location.path("/example/" + idx, true);
        $scope.gotoExampleIdx = idx;

        // Compute everything that's needed to render this example.
        $scope.example = State.getExample(idx - 1);
        $scope.metricValues = $scope.systems.map(system => $scope.example.predictions[system.key].metrics);
    };
    $scope.gotoInputError = false;
    $scope.updateGoto = (function () {
        let timer = null;
        return function () {
            if (timer !== null) {
                $timeout.cancel(timer);
                timer = null;
            }
            // Instant validation.
            let idx = validateIndex($scope.gotoExampleIdx);
            if (idx !== null) {
                $scope.gotoInputError = false;
                // Delayed switch.
                timer = $timeout(() => $scope.switchExample(idx), 500);
            } else {
                $scope.gotoInputError = true;
            }
        };
    })();

    let idx = validateIndex($routeParams.id) || State.lastIndex;
    $scope.switchExample(idx);
}]);

App.controller('CompareCtrl', ['State', '$scope', '$document', function (State, $scope, $document) {
    function fromKeys(array, keys) {
        return keys.map(key => array.find(elem => elem.key === key));
    }

    $scope.systems = State.getSystems();
    $scope.metrics = State.getMetrics().filter(metric => metric.displayInSummary && metric.higherIsBetter !== null);
    $scope.selectedKey = [$scope.systems[0].key, $scope.systems[1].key];
    $scope.selectedSystem = fromKeys($scope.systems, $scope.selectedKey);

    $scope.selectedMetricKeys = [];
    $scope.selectedMetrics = [];
    $scope.filteredExamples = State.getAllExamples();
    $scope.example = null;
    $scope.exampleMetrics = State.getMetrics().filter(metric => metric.displayInExample);


    /* Persistent state */
    let fullyLoaded = false;
    let persistentState = State.getPersistenceState("_compare_ctrl");

    function updatePersistentState() {
        if (!fullyLoaded) return;
        persistentState.selectedKey = $scope.selectedKey;
        persistentState.selectedMetricKeys = $scope.selectedMetricKeys;
        persistentState.currentPage = $scope.pagination.currentPage;
        persistentState.currentIndex = $scope.pagination.currentIndex;
        persistentState.$save();
    }


    /* System selection */
    $scope.validSelection = function () {
        const [a, b] = $scope.selectedKey;
        return a !== null && b !== null && a !== b;
    };
    $scope.swapSelection = function () {
        const [a, b] = $scope.selectedKey;
        // Swap the scope variables here so we don't trigger a change for `validSelection()`.
        // If we don't do this, we'll trigger two digest cycles, each produced by a `dropdown` call. The intermediate
        // state is an invalid state because two choices will be the same.
        $scope.selectedKey = [b, a];
        $scope.selectedSystem = [$scope.selectedSystem[1], $scope.selectedSystem[0]];
        $scope.$$postDigest(() => {
            // This has to run after the digest cycle, because we call `$apply` within `onChange`.
            $("#system-dropdown-A").dropdown("set selected", b);
            $("#system-dropdown-B").dropdown("set selected", a);
        });
    };
    $scope.$watchCollection("selectedSystem", function () {
        if (!$scope.validSelection()) return;
        const [a, b] = $scope.selectedSystem;
        $scope.systemNames = [a.name, b.name];
        $scope.exampleSystemNames = $scope.systemNames.concat(['∆Diff']);
        $scope.metricValues = [a.metrics, b.metrics];
        updateFilteredExamples();
        updatePersistentState();
    });


    /* Example pagination */
    $scope.pagination = {
        examplesPerPage: 10,
        maxNextPages: 4,
        pages: [],
        totalExamples: $scope.filteredExamples.length,
        currentPage: 0,
        currentIndex: 0,
        displayPageRange: [],
        isFirstPage: () => $scope.pagination.currentPage === 0,
        isLastPage: () => $scope.pagination.currentPage + 1 >= $scope.pagination.pages.length,
        isFirstExample: () => $scope.pagination.isFirstPage() && $scope.pagination.currentIndex === 0,
        // jshint ignore: start
        isLastExample: () => $scope.pagination.isLastPage() &&
            $scope.pagination.currentIndex + 1 >= ($scope.pagination.pages.back()?.length ?? 0),
        // jshint ignore: end
        prevExample: () => {
            const pagination = $scope.pagination;
            if (--pagination.currentIndex < 0)
                pagination.currentIndex = pagination.pages[--pagination.currentPage].length - 1;
        },
        nextExample: () => {
            const pagination = $scope.pagination;
            if (++pagination.currentIndex >= pagination.pages[pagination.currentPage].length) {
                ++pagination.currentPage;
                pagination.currentIndex = 0;
            }
        },
    };

    function createPagination() {
        let pagination = $scope.pagination;
        const examplesPerPage = pagination.examplesPerPage;
        pagination.totalExamples = $scope.filteredExamples.length;
        pagination.pages = [];
        for (let idx = 0; idx < pagination.totalExamples; idx += examplesPerPage)
            pagination.pages.push($scope.filteredExamples.slice(idx, idx + examplesPerPage));
        pagination.currentPage = 0;
        pagination.currentIndex = 0;
    }

    createPagination();


    /* Example-related logic */
    function updateFilteredExamples() {
        $scope.selectedMetrics = fromKeys($scope.metrics, $scope.selectedMetricKeys);
        if (!$scope.validSelection()) return;
        const [a, b] = $scope.selectedKey;
        const cmps = State.getAllExamples().map(example => {
            return $scope.selectedMetrics.map(metric => metric.compare(
                example.predictions[a].metrics[metric.key], example.predictions[b].metrics[metric.key]));
        });
        let indices = [...Array(State.getCount()).keys()].filter(idx => cmps[idx].every(x => x < 0));
        // Sort with the comparisons as multiple keywords.
        indices.sort((a, b) => {
            for (let idx in $scope.selectedMetrics)
                if (cmps[a][idx] !== cmps[b][idx]) return cmps[a][idx] - cmps[b][idx];
            return a - b;  // fallback to sort by index
        });
        const allExamples = State.getAllExamples();
        $scope.filteredExamples = indices.map(idx => allExamples[idx]);
        createPagination();
        updateExample();
    }

    function updateExample() {
        if (!$scope.validSelection()) return;

        const pagination = $scope.pagination;
        if (pagination.pages.length === 0) {
            $scope.example = State.getAllExamples()[0];
            pagination.displayPageRange = [];
        } else {
            if (pagination.currentIndex >= pagination.pages[pagination.currentPage].length)
                pagination.currentIndex = pagination.pages[pagination.currentPage].length - 1;
            $scope.example = pagination.pages[pagination.currentPage][pagination.currentIndex];
            const left = Math.max(0, pagination.currentPage - pagination.maxNextPages);
            const right = Math.min(pagination.pages.length - 1, pagination.currentPage + pagination.maxNextPages);
            pagination.displayPageRange = [...Array(right - left + 1).keys()].map(x => x + left);
        }
        updatePersistentState();

        const [a, b] = $scope.exampleMetricValues =
            $scope.selectedKey.map(key => $scope.example.predictions[key].metrics);
        // Compute metric difference.
        $scope.exampleMetricValues.push(Object.fromEntries($scope.selectedMetrics.map(
            metric => [metric.key, metric.difference(a[metric.key], b[metric.key])])));
    }

    updateExample();
    $scope.$watchGroup(["pagination.currentPage", "pagination.currentIndex"], updateExample);


    $scope.metricDropdownColor = function (metric, $index) {
        const cmp = metric.compare($scope.metricValues[0][metric.key], $scope.metricValues[1][metric.key]);
        let colors = ["red", "red"];
        if (cmp <= 0) colors[0] = "green";
        if (cmp >= 0) colors[1] = "green";
        return colors[$index];
    };

    $scope.keyPressed = false;
    $document.unbind('keydown');
    $document.unbind('keyup');
    $document.bind('keydown', (e) => {
        if ($scope.keyPressed) return;
        if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
        const pagination = $scope.pagination;
        $scope.$apply(() => {
            if (e.key === "ArrowLeft") {
                if (!pagination.isFirstExample()) pagination.prevExample();
            } else {
                if (!pagination.isLastExample()) pagination.nextExample();
            }
        });
        $scope.keyPressed = true;
    });
    $document.bind('keyup', (e) => {
        $scope.keyPressed = false;
    });

    $scope.$on('$destroy', function () {
        $document.unbind('keydown');
        $document.unbind('keyup');
    });


    // Load persistent state & initialize Semantic UI DOM elements.
    if (angular.isDefined(persistentState.selectedKey)) {
        $scope.selectedKey = persistentState.selectedKey;
        $scope.selectedSystem = fromKeys($scope.systems, persistentState.selectedKey);
        $scope.selectedMetricKeys = persistentState.selectedMetricKeys;
        // Not done yet; pagination has to be restored after loading metrics.
    } else {
        // Nothing's saved, so we're done loading.
        fullyLoaded = true;
    }
    $scope.$$postDigest(function () {
        // Initialize the system selection dropdowns.
        ["A", "B"].forEach((n, idx) => {
            const $dropdown = $("#system-dropdown-" + n);
            $dropdown.dropdown({
                onChange: (value) => $scope.$apply(() => {
                    $scope.selectedKey[idx] = value;
                    $scope.selectedSystem[idx] = $scope.systems.find(system => system.key === value);
                })
            });
            if ($scope.selectedKey[idx] !== null)
                $dropdown.dropdown("set selected", $scope.selectedKey[idx]);
        });

        // Initialize the metric selection dropdown.
        const $metricDropdown = $("#metric-dropdown");
        // Neither "set exactly" nor "set selected" preserves order when a list is passed. Thus, we simulate
        // selecting one-by-one to manually maintain order.
        for (let key of $scope.selectedMetricKeys)
            $metricDropdown.dropdown("set selected", key);
        updateFilteredExamples();
        // Add `onChange` handler after we apply the initial changes.
        $metricDropdown.dropdown({
            onChange: (value) => $scope.$apply(() => {
                $scope.selectedMetricKeys = value === "" ? [] : value.split(",");
                updateFilteredExamples();
            }),
        });

        if (angular.isDefined(persistentState.currentPage)) {
            $scope.$apply(() => {
                // Pagination is re-created every time metrics change. Wait until the metrics are fully loaded.
                $scope.pagination.currentPage = persistentState.currentPage;
                $scope.pagination.currentIndex = persistentState.currentIndex;
                // Prevent updating persistent state before the pagination is restored.
                fullyLoaded = true;
            });
        }
    });
}]);