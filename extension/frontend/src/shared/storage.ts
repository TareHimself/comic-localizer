import browser from "webextension-polyfill";
export const enum StorageKeys {
    ServerAddress = "serverAddress",
    BatchSize = "batchSize",
    BatchTimeout = "batchTimeout",
    ApiKey = "apiKey",
}

interface IGetStorageOrDefaultsResult {
    [StorageKeys.ServerAddress]: string;
    [StorageKeys.BatchSize]: number;
    [StorageKeys.BatchTimeout]: number;
    [StorageKeys.ApiKey]: string;
}

export const StorageDefaults: IGetStorageOrDefaultsResult = {
    [StorageKeys.ServerAddress]: "http://127.0.0.1:9000/api/v1",
    [StorageKeys.BatchSize]: 4,
    [StorageKeys.BatchTimeout]: 500,
    [StorageKeys.ApiKey]: "",
};

export const getStorageOrDefaults =
    (): Promise<IGetStorageOrDefaultsResult> => {
        return browser.storage.local.get({
            [StorageKeys.ServerAddress]:
                StorageDefaults[StorageKeys.ServerAddress],
            [StorageKeys.BatchSize]: StorageDefaults[StorageKeys.BatchSize],
            [StorageKeys.BatchTimeout]:
                StorageDefaults[StorageKeys.BatchTimeout],
            [StorageKeys.ApiKey]: StorageDefaults[StorageKeys.ApiKey],
        }) as unknown as Promise<IGetStorageOrDefaultsResult>;
    };
